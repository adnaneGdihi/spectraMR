"""Configuration Health Checker.

This module provides comprehensive validation of experiment configurations,
checking for completeness, registry compatibility, and common misconfigurations.

Every health-check failure carries a structured ``category`` /
``yaml_keys`` / ``fix_hint`` triple so the campaign aggregator and the
``python -m spectramr.cli audit`` command can group failures the same way the
hand-curated ``ERRORS_GROUPED_SUMMARY.md`` does — automatically.
"""

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS
from spectramr.config.settings import TrainingSettings
from spectramr.domain.workflows.declaration import (
    declared_regime,
    declared_signal_domain,
    declared_spatial_rank,
    declared_task,
)
from spectramr.infrastructure.physics.dc_settings import (
    inert_dc_knobs,
    resolve_effective_dc,
)
from spectramr.infrastructure.training.utils.metric_transform import (
    IMPLEMENTED_METRIC_TRANSFORMS,
    declared_metric_transforms,
    resolve_metric_transform,
)

# Dataset types where ONE manifest record yields ONE training sample, so
# ``len(records) / batch_size`` bounds the epoch length. Slice-expanding types
# (npy_slice, kspace, ...) produce many samples per record and are excluded —
# guessing there would block a legitimate config.
_ONE_SAMPLE_PER_RECORD_DATASETS = frozenset({"mrixfields", "nifti_paired", "contrast_aware_paired"})

# Under this many iterations per epoch, per-epoch worker respawn stops amortizing.
_SHORT_EPOCH_ITERS = 50

#: The ONE ``data.target_mode`` that complex-averages phase-incoherent M4Raw
#: repetitions and so cancels signal. Named positively, because the alternative
#: -- an allowlist of the acceptable modes -- is a second owner of the schema's
#: ``target_mode`` vocabulary and drifts from it in silence: this module listed
#: ``("phase_aligned_mean", "rep_pair")`` when ``r2r`` joined the Literal, and an
#: r2r arm was consequently told it "declares data.target_mode: complex_mean
#: explicitly", naming a value it does not declare. Every other member is
#: coherent by construction -- ``phase_aligned_mean`` aligns before averaging,
#: ``rep_pair`` takes one already-aligned partner repetition, and ``r2r`` never
#: crosses repetitions at all -- so a mode added to the Literal is coherent
#: unless it is this one.
_INCOHERENT_NEX_TARGET_MODE = "complex_mean"

#: The config paths that decide WHERE one arm's artifacts land. Every one is
#: read on a production path -- ``bootstrap.py`` resolves the log sink and the
#: metrics directory, and ``training.output_dir`` is read at hundreds of sites --
#: which is what makes a disagreement between them a defect a run can act on.
#:
#: ``logging.identity.experiment`` looks like the seventh member and deliberately
#: is not one. Nothing under ``src/`` consumes it: the only two mentions are a
#: ``renames.py`` row and an entry in
#: ``paired_arms_diff_paths.DEFAULT_DIFF_PATHS``, which is the allow-list of
#: paths two paired arms may legitimately DIFFER on. Comparing a path that is
#: written against a label that is never read would report a disagreement no run
#: can act on. It is missing from ``KNOWN_UNCONSUMED`` only because that ledger
#: matches a key's LEAF token and ``experiment`` is spelled all over ``src/``
#: (#1925); the unread knob itself is filed separately, not enforced here.
#:
#: This tuple is the single owner (non-negotiable 17).
#: ``scripts/ci/check_identity_paths_unique.py`` imports it rather than keeping a
#: second copy, so the corpus-wide gate and the per-config check below can never
#: drift about what counts as an identity path.
IDENTITY_PATHS: tuple[str, ...] = (
    "training.output_dir",
    "checkpoint.checkpoint_dir",
    "logging.sinks.dir",
    "loss_logging.output_dir",
    "loss_logging.csv_path",
    "metrics.output_dir",
)

#: The results-root convention :meth:`check_output_dir_convention` enforces on
#: ``training.output_dir``. ``cli/profile_paths.CONVENTION_PARTS`` already
#: derives its copy from this module rather than becoming a second owner.
RESULTS_PREFIX = "experiments/results/"

#: Checks whose failure means the run CANNOT succeed, so the pipeline aborts
#: before ``bootstrap.build_container`` instead of warning and continuing.
#:
#: The admission rule is deliberately narrow, and it is about *certainty*, not
#: severity: an entry may only be added when a failure makes the run impossible,
#: so listing it changes WHERE the run dies and never WHETHER it dies. Both
#: current members are provably that — the builder raises on each regardless, so
#: no arm can be relying on the warn-and-continue path.
#:
#: Every other error-severity check stays non-fatal on purpose. There are ~150 of
#: them and many fire on configs that genuinely train (a wrong-but-runnable
#: claim, a suspicious-but-legal combination); promoting them wholesale would
#: turn this gate into a blanket refusal. Adding one is a per-check judgement
#: that has to answer: does the run *always* die anyway?
FATAL_HEALTH_CHECKS: frozenset[str] = frozenset(
    {
        # Channel/domain mismatch — the model cannot consume what the data
        # pipeline emits. Fail-fast since the audit ladder's introduction.
        "domain_alignment",
        # parallel.strategy='deepspeed' without the [deepspeed] extra. The check
        # message said "the run would fail after building the whole training
        # environment" and the pipeline then did exactly that: on a cluster node
        # the arm validated data, resolved the device and built the environment
        # before DeepSpeedUnavailableError. Nothing recovers an absent import.
        "deepspeed_extra_installed",
    }
)

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a configuration health check.

    The optional ``category`` / ``yaml_keys`` / ``fix_hint`` fields make
    the result machine-readable for the audit aggregator. Older callers
    that only supply ``passed`` / ``check_name`` / ``message`` /
    ``severity`` continue to work.

    ``always_report`` separates *advisory* from *invisible*. ``log_summary``
    renders a result only when it did not pass, so an ``info``-severity check
    that returns ``passed=True`` by design -- the polarity that keeps it from
    gating the audit -- could not put a character in a training log. Setting
    this flag emits the message at ``info`` without touching ``report.passed``
    or ``report.warnings``, so the check stays non-gating and becomes legible.
    """

    passed: bool
    check_name: str
    message: str
    severity: str = "warning"  # "warning", "error", "info"
    # Structured-diagnostic fields (audit ladder spec, 2026-05-03):
    category: str | None = None
    yaml_keys: list[str] = field(default_factory=list)
    fix_hint: str | None = None
    # Opt-in, and deliberately NOT `severity == "info"`: 140 of the 141 results
    # on a real arm are passing info results, and 16 of those carry a category,
    # nearly all of them "not applicable" / "check skipped". Gating on either
    # would bury the findings in the confirmations, which is the failure mode
    # this field exists to avoid. A check sets it on the branch that states a
    # finding, never on its n/a branches.
    always_report: bool = False

    def __str__(self) -> str:
        """__str__.

        Returns:
            str: Description.
        """
        icon = "✅" if self.passed else ("❌" if self.severity == "error" else "⚠️")
        suffix = ""
        if not self.passed and self.fix_hint:
            suffix = f"\n     fix: {self.fix_hint}"
        return f"{icon} [{self.check_name}] {self.message}{suffix}"

    def __rich__(
        self,
    ) -> "Text":  # noqa: F821  (rich.text.Text — string-quoted for lazy import)
        """Rich-aware rendering. Falls back to plain ``__str__`` on non-TTY consoles.

        Round-11 (2026-05-17): adds ANSI color when stdout is a TTY so the
        ``python -m spectramr.cli audit`` output is readable at a glance on the
        cluster (and locally). When piped / non-TTY, rich strips styles and
        the rendered output is identical to ``__str__`` — CI / grep / JSON
        consumers are unaffected.
        """
        from rich.text import Text

        if self.passed:
            icon, icon_style = "✅", "green"
        elif self.severity == "error":
            icon, icon_style = "❌", "bold red"
        else:
            icon, icon_style = "⚠", "yellow"
        text = Text()
        text.append(f"{icon} ", style=icon_style)
        text.append(f"[{self.check_name}] ", style="cyan")
        text.append(self.message)
        if not self.passed and self.fix_hint:
            text.append("\n     fix: ", style="dim")
            text.append(self.fix_hint, style="italic")
        return text

    def to_dict(self) -> dict:
        """Return a JSON-serialisable view of the result."""
        return asdict(self)


@dataclass
class HealthCheckReport:
    """Aggregated health check results."""

    results: list[HealthCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Overall pass status (no errors)."""
        return not any(r.severity == "error" and not r.passed for r in self.results)

    @property
    def warnings(self) -> list[HealthCheckResult]:
        """Get all warnings."""
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    @property
    def errors(self) -> list[HealthCheckResult]:
        """Get all errors."""
        return [r for r in self.results if not r.passed and r.severity == "error"]

    def log_summary(self) -> None:
        """Log a summary of the health check.

        Non-passing results are rendered at their severity, and passing ones
        that set ``always_report`` at ``info``. Neither branch reads or writes
        the aggregate verdicts -- ``passed`` / ``warnings`` / ``errors`` count
        only non-passing results, so making an advisory visible here cannot
        change any surface's exit code.
        """
        passed_count = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        logger.info(f"Config Health: {passed_count}/{total} checks passed")

        for result in self.results:
            if not result.passed:
                log_fn = logger.error if result.severity == "error" else logger.warning
                log_fn(str(result))
            elif result.always_report:
                # A passing advisory. Emitted at `info` -- the same level as the
                # `Config Health: n/m` line above, so wherever the summary
                # survives the sink's level clamp this does too -- and read by
                # nothing that decides pass/fail.
                logger.info(str(result))

    def to_dict(self) -> dict:
        """Return the report as a JSON-serialisable dict.

        Schema:
            ``{"passed": bool, "n_errors": int, "n_warnings": int,
              "results": [HealthCheckResult.to_dict(), ...]}``
        """
        return {
            "passed": self.passed,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Return the report serialised to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


def _has_enabled_pre_model_adapter(adapters: Any) -> bool:
    """True when ``adapters`` declares at least one ENABLED ``pre_model`` step.

    The narrow form of "an adapter may bridge the data domain". Only ``pre_model`` sits
    between the dataset and the model, so only it can change what domain the model
    receives; the other four hooks (``post_model``, ``pre_loss_pred``,
    ``pre_loss_target``, ``pre_metric``) run after the model has already consumed the
    input. A step with ``enabled: false`` is declared-but-skipped and bridges nothing.

    Tolerates a test double (``SimpleNamespace``, a plain dict) as well as the real
    ``AdaptersConfigSchema``, and treats "cannot tell" as "no adapter", so an
    unrecognised shape leaves the check ARMED rather than silently disarming it.
    """
    if adapters is None:
        return False
    steps = (
        adapters.get("pre_model")
        if isinstance(adapters, dict)
        else getattr(adapters, "pre_model", None)
    )
    if not steps:
        return False
    for step in steps:
        enabled = (
            step.get("enabled", True) if isinstance(step, dict) else getattr(step, "enabled", True)
        )
        if enabled:
            return True
    return False


def _contrast_aware_critics(registry: dict[str, Any]) -> tuple[list[str], int]:
    """Registered discriminator names, and which of them declare contrast support.

    Returns ``(aware_names, total_names)``. There is no longer a ``None`` arm:
    it existed to report an unimportable ``IDiscriminator`` as an *unknown*
    denominator rather than a constant (non-negotiable 18), and reading the
    declared role imports nothing, so the absence it guarded cannot occur.

    This exists because the sentence it feeds used to be hard-coded as
    "0 of 21". No registry has ever held 21: when that constant was written
    ``MODEL_REGISTRY`` carried 14 discriminator *names* (``patch_gan`` and
    ``patchgan_discriminator`` being two names for one class) and
    ``ModelFactory``'s separate registry carried 13. So it was already wrong,
    and it could not self-correct: the day a critic declares the flag, the
    check stops firing on that arm, and the now-false sentence stays on every
    *other* arm with nobody reading it.

    That day has since arrived -- ``sense_bridge_patchgan_conditioned`` declares
    the flag (#1931 half 2), moving the census off zero -- which is the whole
    argument for recomputing rather than quoting. No number is repeated in this
    docstring for the same reason; run the function.

    A name counts as a critic if it was **registered** as one -- ``role=
    "discriminator"`` on ``register_model``. That replaces an
    ``IDiscriminator``-subclass-or-module-path union (#1916/#1932): the union
    was a second resolver for "is this a critic", guessing from two unrelated
    properties what registration can simply state, and this function then
    disagreed with ``ModelRegistry``'s own bucketing about the denominator it
    publishes. Measured at the swap: the union answered **15**, the declared
    role answers **21**, and the union is a strict subset (0 names the role
    misses), so this also satisfies the criterion the union was chosen for --
    "none of them" asserted over the *widest* candidate set, not the narrowest.
    The 6 it adds are critics no heuristic could see: they neither subclass
    ``IDiscriminator`` nor ship from a ``discriminators`` module
    (``ldm_*_latent_discriminator``, ``wasserstein_discriminator``,
    ``domain_discriminator``, ``kan_discriminator``).

    The count is over names, not classes, because names are what a YAML writes.
    """
    critics = [name for name, entry in registry.items() if entry.get("role") == "discriminator"]
    # Read the NESTED capability, not a top-level key (#1916). ``register_model``
    # no longer fans conditioning flags out to the entry dict -- there is one
    # owner now -- so a ``registry[n].get(...)`` here would answer False for
    # every critic and print "0 aware" forever, which is the precise silent
    # zero this function was written to stop quoting. ``getattr`` on the
    # capabilities object keeps a hand-built fake entry (no ``capabilities``
    # key) answering False rather than raising.
    aware = sorted(
        n
        for n in critics
        if getattr(registry[n].get("capabilities"), "supports_contrast_conditioning", None)
    )
    return aware, len(critics)


def _pins_materialised_agreement(losses: Any, site: str) -> bool:
    """Read a ``LossWeightSpec.source`` segment as a field and ask the weight owner.

    An unrecognised segment shape answers *yes*. The caller's only use of a
    ``False`` is to authorise deleting the field, and a source format this cannot
    parse is not evidence the deletion is safe (#9).
    """
    from spectramr.models.losses.weights import deleting_lambda_would_conflict

    parts = site.split(".")
    if len(parts) != 3 or parts[0] != "losses":
        return True
    return deleting_lambda_would_conflict(losses, parts[1], parts[2])


def dual_surface_loss_declarations(losses: Any) -> list[tuple[str, float, list[str], list[str]]]:
    """Loss weights written on the lambda surface *and* a domain list.

    Sole owner of the pairing rule (non-negotiable 17): the audit check and
    ``scripts/migrations/migrate_loss_lambdas_to_domain_lists.py`` both read it, so
    the migration can never delete a field the audit would not have flagged. The
    migration uses it as a *gate*, not as its candidate source -- it scans the YAML
    text for candidates and drops every one this function does not report, because
    the reverse direction is not safe: the canonicalization below means a pair this
    function cannot represent at all (``losses.diffusion.lambda_mse`` resolves onto
    ``l2``) would otherwise read as "nothing to keep".

    Returns ``(canonical_name, weight, lambda_sites, list_sites)`` per duplicated
    weight. Detection is the weight table's own provenance -- ``LossWeightSpec.source``
    is ``"+"``-joined when a canonical loss was declared more than once -- so no
    second reader of either surface is introduced.

    Three exclusions, all deliberate:

    * a ``source`` naming only lists (``image_losses`` + ``kspace_losses``) is a
      different question, owned by ``check_loss_domain_consistency``;
    * a lambda in ``REINTERPRETED_LAMBDA_SOURCES`` is not redundant. It is
      read by a computer under a *different* meaning, so deleting it removes a
      knob rather than a duplicate. That is a narrower set than
      ``COMPUTER_RESOLVED_LAMBDA_SOURCES``, which also holds the lambdas a
      computer reads straight off the table under the same meaning a list entry
      gives them (``reconstruction.lambda_l1`` and three more) — those ARE
      duplicates beside a list entry and belong in this report. ``losses.diffusion.lambda_mse`` is step 3 of
      ``UnifiedDiffusionReconstructionLossComputer._resolve_diffusion_weight``
      (the diffusion-term weight), while the table aliases ``lambda_mse`` ->
      ``l2`` and so joins it to any image/k-space ``mse`` entry. One table key,
      two knobs;
    * a lambda that PINS the materialised view (``deleting_lambda_would_conflict``)
      is not redundant either, for an unrelated reason: the written surface shows
      a duplicate, but the value is what holds two sections' divergent schema
      defaults equal, so removing it makes the two disagree. The two
      exclusions overlap on ``lambda_l2`` today and are kept apart because they
      would survive each other — rewiring the diffusion computer would retire the
      second, and issue #421 would retire the third.
    """
    from spectramr.infrastructure.training.builders.loss_builder import (
        REINTERPRETED_LAMBDA_SOURCES,
    )
    from spectramr.models.losses.weights import build_loss_weight_table

    table = build_loss_weight_table(losses)
    duplicated: list[tuple[str, float, list[str], list[str]]] = []
    for spec in table.values():
        # Deliberately NOT filtered on ``spec.enabled``: a list entry with
        # ``enabled: false`` beside a positive lambda resolves to weight 0 with
        # no disagreement raised, which is exactly the shape where the two
        # surfaces are least obviously one term.
        segments = spec.source.split("+")
        lambda_sites = [
            s
            for s in segments
            if ".lambda_" in s
            and s not in REINTERPRETED_LAMBDA_SOURCES
            and not _pins_materialised_agreement(losses, s)
        ]
        list_sites = [s for s in segments if "_losses[" in s]
        if lambda_sites and list_sites:
            duplicated.append((spec.name, spec.weight, lambda_sites, list_sites))
    return sorted(duplicated)


class ConfigHealthChecker:
    """Validates configuration completeness and compatibility.

    Checks include:
    - Required fields presence
    - Model type registry lookup
    - Strategy registry compatibility
    - Loss weight sanity checks
    - Physics configuration validation
    """

    # Required top-level config sections (actual TrainingSettings fields)
    REQUIRED_SECTIONS = ["data", "model", "training", "optimization", "logging"]

    def __init__(self) -> None:
        """__init__."""
        self._model_registry: set[str] | None = None
        self._strategy_registry: set[str] | None = None
        self._loss_registry: set[str] | None = None

    def _lazy_load_registries(self) -> None:
        """Lazily load model/strategy/loss registries to avoid import cycles."""
        if self._model_registry is not None:
            return

        try:
            # Use full ModelFactory (includes manual registrations + decorator-based)
            # ModelRegistry alone only has ~55 decorator entries, but ModelFactory
            # adds 136+ more via _init_generators() for a total of ~191.
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                from spectramr.models.factories.model_factory import ModelFactory

                factory = ModelFactory()
                inner_reg = factory._model_creator._registry
                self._model_registry = (
                    set(inner_reg._generators.keys())
                    if hasattr(inner_reg, "_generators")
                    else set()
                )
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning("Model registry unavailable: %s", e)
            self._model_registry = set()
        except AttributeError as e:
            logger.warning("ModelFactory registry access failed: %s", e)
            self._model_registry = set()

        try:
            # [FIX] Use TrainingStrategyFactory registry (single source of truth)
            from spectramr.infrastructure.training.strategy_factory import (
                TrainingStrategyFactory,
            )

            self._strategy_registry = set(TrainingStrategyFactory.STRATEGY_CLASS_PATHS.keys())
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning("Strategy registry unavailable: %s", e)
            self._strategy_registry = set()
        except AttributeError as e:
            logger.warning("TrainingStrategyFactory missing CLASS_PATHS: %s", e)
            self._strategy_registry = set()

        try:
            # Importing the package triggers every @register_loss decorator.
            import spectramr.models.losses  # noqa: F401
            from spectramr.models.losses.registry import LossRegistry

            canonical = set(LossRegistry._custom_losses.keys())
            aliases = set(LossRegistry._aliases.keys())
            self._loss_registry = {n.lower() for n in (canonical | aliases)}
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            logger.warning("Loss registry unavailable: %s", e)
            self._loss_registry = set()

    def check_required_sections(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Check that required configuration sections are present."""
        results = []

        for section in self.REQUIRED_SECTIONS:
            has_section = hasattr(config, section) and getattr(config, section) is not None
            results.append(
                HealthCheckResult(
                    passed=has_section,
                    check_name="required_section",
                    message=f"Section '{section}' {'present' if has_section else 'MISSING'}",
                    severity="error" if not has_section else "info",
                )
            )

        return results

    def check_model_registry(self, config: TrainingSettings) -> HealthCheckResult:
        """Verify model_type exists in registry."""
        self._lazy_load_registries()

        model_type = config.model.model_type
        if model_type is None:
            return HealthCheckResult(
                passed=False,
                check_name="model_registry",
                message="model.model_type not specified",
                severity="error",
            )

        # An empty registry means the check could not run — it does NOT mean the
        # model is fine. Reporting it as passed/info turned the audit's single
        # model-existence gate into a no-op whenever the import broke, so a
        # typo'd model_type sailed through pre-flight and failed at build time
        # instead (non-negotiable 3: absent is a state to report, never a state
        # to infer). ``_lazy_load_registries`` only empties the set on an
        # ImportError/AttributeError, both already logged, so this fires on a
        # genuine import regression and nothing else — measured on this host:
        # 332 model names / 206 strategy keys load.
        if not self._model_registry:
            return HealthCheckResult(
                passed=False,
                check_name="model_registry",
                message=(
                    f"model registry unavailable — cannot validate "
                    f"model_type='{model_type}'. This is an import regression in "
                    "ModelFactory, not a property of the config; see the "
                    "'Model registry unavailable' warning logged above."
                ),
                severity="error",
            )

        in_registry = model_type in self._model_registry
        return HealthCheckResult(
            passed=in_registry,
            check_name="model_registry",
            message=f"model_type='{model_type}' {'registered' if in_registry else 'NOT FOUND in registry'}",
            severity="error" if not in_registry else "info",
        )

    def check_registered_model_resolves(self, config: TrainingSettings) -> HealthCheckResult:
        """Tier-1: a registered model_type must resolve to a *concrete* class.

        ``check_model_registry`` already verifies membership. This adds the
        regression guard for the Category-C re-introduction failure mode
        (TODO/deleted_model_types_reimplementation_plan.md Phase 0): an
        advertised name that maps to an abstract base class, ``None``, or a
        class without a ``forward`` would pass Tier-0 yet ``KeyError`` / raise
        ``TypeError`` at ``model_factory.build()``. We inspect the decorator
        registry (``MODEL_REGISTRY``) because it carries the class objects;
        names that exist only in the wider ModelFactory set are skipped here
        (they are covered by ``check_model_registry``).
        """
        import inspect

        model_type = config.model.model_type
        if not model_type:
            return HealthCheckResult(
                passed=True,
                check_name="registered_model_resolves",
                message="no model_type to resolve",
                severity="info",
            )
        try:
            from spectramr.models.init_registry import populate_model_registry
            from spectramr.models.registry import MODEL_REGISTRY

            if not MODEL_REGISTRY:
                populate_model_registry()
        except Exception as e:
            # Same posture as check_model_registry above: a registry that will
            # not import is a failure of the checker's precondition, and a
            # silently-skipped Tier-1 resolve check is indistinguishable from a
            # resolve check that passed.
            return HealthCheckResult(
                passed=False,
                check_name="registered_model_resolves",
                message=(
                    f"model registry unavailable ({e}); the resolve-check could "
                    "NOT run — this is an import regression, not a clean config"
                ),
                severity="error",
            )

        entry = MODEL_REGISTRY.get(model_type)
        if entry is None:
            # Not in the decorator registry; membership is enforced by
            # check_model_registry against the wider factory set.
            return HealthCheckResult(
                passed=True,
                check_name="registered_model_resolves",
                message=f"model_type='{model_type}' not in decorator registry (deferred to model_registry check)",
                severity="info",
            )

        cls = entry.get("class")
        problems: list[str] = []
        if cls is None:
            problems.append("registry entry has no class object")
        else:
            if inspect.isabstract(cls):
                problems.append(
                    f"{cls.__qualname__} is abstract "
                    f"({sorted(getattr(cls, '__abstractmethods__', ()))})"
                )
            if not callable(getattr(cls, "forward", None)):
                problems.append(f"{cls.__qualname__} has no callable forward()")

        if problems:
            return HealthCheckResult(
                passed=False,
                check_name="registered_model_resolves",
                message=(
                    f"model_type='{model_type}' resolves to a non-buildable class: "
                    + "; ".join(problems)
                ),
                severity="error",
                yaml_keys=["model.model_type"],
                category="registered_model_resolves",
                fix_hint=(
                    "Register a concrete nn.Module (with forward) under this "
                    "name, or remove the name from VALID_MODEL_TYPES."
                ),
            )
        return HealthCheckResult(
            passed=True,
            check_name="registered_model_resolves",
            message=f"model_type='{model_type}' → concrete buildable class",
            severity="info",
        )

    @staticmethod
    def _validation_sample_is_a_stub(cls: type) -> bool:
        """True when ``cls.validation_sample`` can only raise ``NotImplementedError``.

        A method that raises on *some* path and returns on another is a live
        implementation with a guard; one that carries a ``raise NotImplementedError``
        and no ``return``/``yield`` anywhere cannot produce a reconstruction on any
        input. Verified to separate the four adapters on ``dev``: the base and
        ``FDBBaseline`` are stubs, ``CDiffMRBaseline`` and ``Shen2024Baseline`` are not.
        """
        import ast
        import inspect
        import textwrap

        fn = None
        for klass in getattr(cls, "__mro__", (cls,)):
            if "validation_sample" in vars(klass):
                fn = vars(klass)["validation_sample"]
                break
        if fn is None:
            return False
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        except (OSError, TypeError, SyntaxError, IndentationError):
            # No source (C extension, exec'd class): unprovable, so not asserted.
            return False

        nodes = list(ast.walk(tree))
        returns_a_value = any(
            isinstance(n, ast.Return) and n.value is not None for n in nodes
        ) or any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in nodes)
        raises_not_implemented = any(
            isinstance(n, ast.Raise)
            and n.exc is not None
            and (
                (isinstance(n.exc, ast.Call) and getattr(n.exc.func, "id", "") == "NotImplementedError")
                or getattr(n.exc, "id", "") == "NotImplementedError"
            )
            for n in nodes
        )
        return raises_not_implemented and not returns_a_value

    def check_validation_path_is_wired(self, config: TrainingSettings) -> HealthCheckResult:
        """Tier-1: an arm whose validation drives a reverse process must have one.

        ``UpstreamProcessStrategy`` validates through the adapter's own
        ``validation_sample`` rather than a bare ``forward``. When that method is a
        stub, every validation batch raises, ``_run_validation`` correctly refuses to
        call the run passing — and all of it happens at the FIRST eval interval, after
        the training budget up to that point has already been spent.

        ``baseline_fdb`` cost 999 iterations and 18 minutes of GPU that way on
        2026-09-21 while ``audit --strict`` reported it clean (#2255). Its three
        sibling baselines declare ``metadata.status='needs_implementation'`` and are
        refused in under two seconds; this check gives the same answer to an arm whose
        gap is in the code rather than in its metadata.

        Declaring the status is the accepted way to run one anyway: the launch guard
        then names it, and ``--allow-status needs_implementation`` is an explicit
        wiring exercise rather than an accident.
        """
        model_type = getattr(getattr(config, "model", None), "model_type", "") or ""
        if not model_type:
            return HealthCheckResult(
                passed=True,
                check_name="validation_path_is_wired",
                message="no model_type to resolve",
                severity="info",
            )

        from spectramr.config.schemas.base import LAUNCH_REFUSED_STATUSES

        status = getattr(getattr(config, "metadata", None), "status", None)
        if status in LAUNCH_REFUSED_STATUSES:
            return HealthCheckResult(
                passed=True,
                check_name="validation_path_is_wired",
                message=f"metadata.status={status!r} already refuses this arm at launch",
                severity="info",
            )

        try:
            from spectramr.models.init_registry import populate_model_registry
            from spectramr.models.registry import MODEL_REGISTRY

            if not MODEL_REGISTRY:
                populate_model_registry()
        except Exception as e:
            return HealthCheckResult(
                passed=False,
                check_name="validation_path_is_wired",
                message=(
                    f"model registry unavailable ({e}); the validation-path check "
                    "could NOT run — an import regression, not a clean config"
                ),
                severity="error",
            )

        entry = MODEL_REGISTRY.get(model_type)
        cls = entry.get("class") if entry else None
        if cls is None or not self._validation_sample_is_a_stub(cls):
            return HealthCheckResult(
                passed=True,
                check_name="validation_path_is_wired",
                message=f"model_type='{model_type}' has no stubbed validation_sample",
                severity="info",
            )

        return HealthCheckResult(
            passed=False,
            check_name="validation_path_is_wired",
            message=(
                f"model_type='{model_type}' ({cls.__qualname__}.validation_sample) can "
                "only raise NotImplementedError, so EVERY validation batch will fail and "
                "the run will abort at the first eval interval — after spending the "
                "training budget up to that point"
            ),
            severity="error",
            yaml_keys=["model.model_type", "metadata.status"],
            category="validation_path_is_wired",
            fix_hint=(
                "Implement validation_sample, or declare "
                "metadata.status='needs_implementation' with a status_reason so the arm "
                "is refused at launch like its siblings (run it deliberately with "
                "--allow-status needs_implementation)."
            ),
        )

    def check_namespace_axis(self, config: TrainingSettings) -> HealthCheckResult:
        """Tier-1: reject cross-axis token reuse under ``model.model_type``.

        The original deletion ledger (commit d8ccb8452) found training_mode
        tokens (``gan``), dataset-type tokens (``dicom_series``, ``2d_stack``),
        strategy keys (``test_time_adaptation_strategy``), and block names
        (``photonic_fft``, ``sparse_transformer``) misfiled as ``model_type``.
        Re-adding those as models is what made the v6.0 schema drift. This
        check fires ONLY when the name fails to resolve as a model but DOES
        resolve on another axis — giving a precise fix hint instead of the
        generic "NOT FOUND in registry". A name that resolves as a model is
        always accepted (so legitimate models that happen to share a string
        with a block are never blocked).
        """
        self._lazy_load_registries()
        model_type = config.model.model_type
        if not model_type:
            return HealthCheckResult(
                passed=True,
                check_name="namespace_axis",
                message="no model_type to lint",
                severity="info",
            )

        # If it resolves as a model, it is correctly filed — never flag.
        try:
            from spectramr.models.init_registry import populate_model_registry
            from spectramr.models.registry import MODEL_REGISTRY

            if not MODEL_REGISTRY:
                populate_model_registry()
            resolves_as_model = model_type in MODEL_REGISTRY or model_type in self._model_registry
        except Exception:
            resolves_as_model = model_type in self._model_registry
        if resolves_as_model:
            return HealthCheckResult(
                passed=True,
                check_name="namespace_axis",
                message=f"model_type='{model_type}' resolves as a model",
                severity="info",
            )

        # Explicitly-rejected aspirational names (Phase 4) get the curated
        # rejection rationale as the fix hint.
        try:
            from spectramr.models.registry import REJECTED_NAMES

            if model_type in REJECTED_NAMES:
                return HealthCheckResult(
                    passed=False,
                    check_name="namespace_axis",
                    message=(
                        f"model_type='{model_type}' is a rejected aspirational "
                        f"name: {REJECTED_NAMES[model_type]}"
                    ),
                    severity="error",
                    yaml_keys=["model.model_type"],
                    category="namespace_axis",
                    fix_hint=(
                        f"'{model_type}' was rejected ({REJECTED_NAMES[model_type]}). "
                        "Pick an implemented model_type or file a real "
                        "implementation under a specific paper's name."
                    ),
                )
        except Exception:
            pass

        # Assemble the foreign-axis token sets (best effort; each guarded).
        training_modes = {
            "gan",
            "diffusion",
            "vae",
            "vqvae",
            "reconstruction",
            "ssl",
            "cycle_gan",
            "domain_adaptation",
            "flow_matching",
            "generative",
            "adversarial_robustness",
        }
        strategy_tokens = set(self._strategy_registry or set())
        dataset_tokens: set[str] = set()
        try:
            from spectramr.config.schemas.enums import DatasetType

            dataset_tokens = {str(getattr(m, "value", m)) for m in DatasetType}
        except Exception:
            pass
        block_tokens: set[str] = set()
        try:
            from spectramr.models.blocks import BLOCK_REGISTRY

            block_tokens = set(BLOCK_REGISTRY.keys())
        except Exception:
            pass

        axis_hits = []
        if model_type in training_modes:
            axis_hits.append("training.training_mode")
        if model_type in strategy_tokens:
            axis_hits.append("training.strategy_class / training.training_mode")
        if model_type in dataset_tokens:
            axis_hits.append("data.dataset_type")
        if model_type in block_tokens:
            axis_hits.append("model.model_kwargs (block name, not a model)")

        if axis_hits:
            return HealthCheckResult(
                passed=False,
                check_name="namespace_axis",
                message=(
                    f"model_type='{model_type}' is a {', '.join(axis_hits)} token "
                    f"misfiled under model.model_type"
                ),
                severity="error",
                yaml_keys=["model.model_type"],
                category="namespace_axis",
                fix_hint=(
                    f"'{model_type}' belongs to another config axis. Move it to "
                    f"the correct field: {', '.join(axis_hits)}."
                ),
            )
        # Unresolved but not a known foreign token → leave to model_registry.
        return HealthCheckResult(
            passed=True,
            check_name="namespace_axis",
            message=f"model_type='{model_type}' not a known foreign-axis token",
            severity="info",
        )

    def check_phase3_model_constraints(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Tier-1 per-model cross-field checks for the Phase-3 models.

        Mirrors the constructor-time validation of the 12 re-implemented
        models (TODO/phase3_model_implementations_detailed.md §*.4) so an
        ill-formed ``model.model_kwargs`` fails at ~100 ms audit time instead
        of at model-build / forward time. Only the active ``model_type`` is
        examined; everything else is a single info pass.
        """
        results: list[HealthCheckResult] = []
        mt = getattr(config.model, "model_type", None)
        kw = dict(getattr(config.model, "model_kwargs", {}) or {})

        def err(msg: str, hint: str) -> None:
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="phase3_model_constraints",
                    message=f"[{mt}] {msg}",
                    severity="error",
                    yaml_keys=["model.model_kwargs"],
                    category="phase3_model_constraints",
                    fix_hint=hint,
                )
            )

        def warn(msg: str, hint: str) -> None:
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="phase3_model_constraints",
                    message=f"[{mt}] {msg}",
                    severity="warning",
                    yaml_keys=["model.model_kwargs"],
                    category="phase3_model_constraints",
                    fix_hint=hint,
                )
            )

        def _patch_side() -> int | None:
            ps = getattr(
                getattr(getattr(config, "data", None), "sampling", None),
                "patch_size",
                None,
            )
            if isinstance(ps, (list, tuple)) and ps:
                try:
                    return int(ps[0])
                except (TypeError, ValueError):
                    return None
            if isinstance(ps, int):
                return ps
            return None

        if mt == "glow":
            num_scales = int(kw.get("num_scales", 3))
            if num_scales < 1:
                err("num_scales must be >= 1", "Set model_kwargs.num_scales >= 1.")
            side = _patch_side()
            if side is not None and num_scales >= 1 and side % (2**num_scales) != 0:
                err(
                    f"patch side {side} not divisible by 2**num_scales (={2**num_scales})",
                    "Pad/crop data.sampling.patch_size to a multiple of 2**num_scales, or lower num_scales.",
                )
        elif mt == "equivariant_flow":
            order = int(kw.get("group_order", 8))
            hidden = kw.get("hidden_channels")
            if order < 1:
                err("group_order must be >= 1", "Set model_kwargs.group_order >= 1.")
            if hidden is not None and int(hidden) % order != 0:
                err(
                    f"hidden_channels ({hidden}) not divisible by group_order ({order})",
                    "Make hidden_channels a multiple of group_order.",
                )
        elif mt == "divergence_free_flow":
            ndim = int(kw.get("ndim", 2))
            if ndim not in (2, 3):
                err("ndim must be 2 or 3", "Set model_kwargs.ndim to 2 or 3.")
        elif mt == "blurring_diffusion":
            if int(kw.get("num_timesteps", 1000)) < 1:
                err("num_timesteps must be >= 1", "Set model_kwargs.num_timesteps >= 1.")
            if float(kw.get("sigma_min", 0.01)) > float(kw.get("sigma_max", 0.1)):
                err("sigma_min must be <= sigma_max", "Swap/adjust the sigma bounds.")
        elif mt == "hierarchical_vae_ladder":
            ld = kw.get("latent_dims")
            nl = kw.get("num_layers")
            if isinstance(ld, (list, tuple)) and nl is not None and len(ld) != int(nl):
                err(
                    f"len(latent_dims)={len(ld)} != num_layers={nl}",
                    "Make latent_dims have exactly num_layers entries.",
                )
        elif mt == "hyperspherical_vae":
            d = kw.get("latent_dim")
            if d is not None and not (2 <= int(d) <= 64):
                warn(
                    f"latent_dim={d} outside [2, 64]; vMF Bessel-ratio numerics "
                    "degrade outside this range",
                    "Prefer latent_dim in [2, 64].",
                )
        elif mt == "moe_vae":
            ne = kw.get("num_experts")
            if ne is not None and int(ne) < 2:
                err("num_experts must be >= 2", "Set model_kwargs.num_experts >= 2.")
        elif mt == "progressive_gan":
            mr = int(kw.get("max_resolution", 256))
            if mr < 4 or (mr & (mr - 1)) != 0:
                err(
                    f"max_resolution ({mr}) must be a power of two >= 4",
                    "Set model_kwargs.max_resolution to 4, 8, 16, ... 256, 512.",
                )
        elif mt == "octave_conv":
            a = float(kw.get("alpha", 0.5))
            if not (0.0 <= a <= 1.0):
                err("alpha must be in [0, 1]", "Set model_kwargs.alpha in [0, 1].")
            elif a in (0.0, 1.0):
                warn(
                    f"alpha={a} degenerates octave conv to a plain/half-res conv",
                    "Use 0 < alpha < 1 to exercise the octave decomposition.",
                )
        elif mt == "gated_gnn":
            steps = int(kw.get("num_steps", 5))
            if steps < 1:
                err("num_steps must be >= 1", "Set model_kwargs.num_steps >= 1.")
            elif steps > 10:
                warn(
                    f"num_steps={steps} > 10 may exhaust memory on dense k-space graphs",
                    "Consider num_steps <= 10.",
                )
            if int(kw.get("num_edge_types", 4)) < 1:
                err(
                    "num_edge_types must be >= 1",
                    "Set model_kwargs.num_edge_types >= 1.",
                )

        if not results:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="phase3_model_constraints",
                    message=f"model_type='{mt}' has no Phase-3 constraint violations",
                    severity="info",
                )
            )
        return results

    def check_strategy_registry(self, config: TrainingSettings) -> HealthCheckResult:
        """Verify strategy_class is specified and valid."""
        self._lazy_load_registries()

        strategy_class = config.training.strategy_class if config.training else None
        training_mode = getattr(config.training, "training_mode", None) if config.training else None

        # training_mode is the canonical selector (STRATEGY_REGISTRY lookup in train.py)
        # strategy_class is optional — only needed for custom/explicit strategies
        if strategy_class is None and training_mode:
            if not self._strategy_registry:
                return HealthCheckResult(
                    passed=True,
                    check_name="strategy_registry",
                    message=f"training_mode='{training_mode}' (registry not available for validation)",
                    severity="info",
                )
            if training_mode in self._strategy_registry:
                return HealthCheckResult(
                    passed=True,
                    check_name="strategy_registry",
                    message=f"training_mode='{training_mode}' → valid STRATEGY_REGISTRY short name",
                    severity="info",
                )
            return HealthCheckResult(
                passed=False,
                check_name="strategy_registry",
                message=(
                    f"training_mode='{training_mode}' NOT FOUND in STRATEGY_REGISTRY. "
                    "TrainingStrategyFactory.get_strategy_class will raise ConfigurationError at runtime."
                ),
                severity="error",
                yaml_keys=["training.training_mode"],
                category="strategy_registry",
                fix_hint=(
                    f"Use one of: {sorted(self._strategy_registry)[:8]}... "
                    "or specify training.strategy_class with the full dotted class path."
                ),
            )

        if strategy_class is None and not training_mode:
            return HealthCheckResult(
                passed=False,
                check_name="strategy_registry",
                message="Neither training.strategy_class nor training.training_mode specified",
                severity="error",
            )

        # If registry is empty, skip check
        if not self._strategy_registry:
            return HealthCheckResult(
                passed=True,
                check_name="strategy_registry",
                message=f"strategy='{strategy_class}' (registry not available)",
                severity="info",
            )

        # Short-name lookup first.
        if strategy_class in self._strategy_registry:
            return HealthCheckResult(
                passed=True,
                check_name="strategy_registry",
                message=f"strategy='{strategy_class}' → valid registry short name",
                severity="info",
            )

        # Full dotted path: actually import-check it. The silent-fallback
        # bug this guards against: an audit that passed because the path
        # *looked* valid (contained a dot), then the pipeline blew up at
        # launch with `module 'foo' has no attribute 'BarStrategy'`. The
        # ldm_two_stage_ulf_to_hf cluster smoke run (2026-05-04, 2026-05-05)
        # is the prototype regression. Cost: ~50ms per audit.
        if "." in strategy_class:
            import importlib

            try:
                module_path, class_name = strategy_class.rsplit(".", 1)
                module = importlib.import_module(module_path)
                if not hasattr(module, class_name):
                    return HealthCheckResult(
                        passed=False,
                        check_name="strategy_registry",
                        message=(
                            f"strategy='{strategy_class}' module imports but "
                            f"class '{class_name}' is NOT FOUND in '{module_path}'."
                        ),
                        severity="error",
                        yaml_keys=["training.strategy_class"],
                        fix_hint=(
                            f"Pick a valid class from "
                            f"spectramr.infrastructure.training.strategies.{module_path.rsplit('.', 1)[-1]} "
                            "or use a short name from STRATEGY_CLASS_PATHS."
                        ),
                    )
                return HealthCheckResult(
                    passed=True,
                    check_name="strategy_registry",
                    message=f"strategy='{strategy_class}' → import-check passed",
                    severity="info",
                )
            except ImportError as e:
                return HealthCheckResult(
                    passed=False,
                    check_name="strategy_registry",
                    message=f"strategy='{strategy_class}' module import FAILED: {e}",
                    severity="error",
                    yaml_keys=["training.strategy_class"],
                    fix_hint="Check the dotted module path; module does not exist.",
                )

        return HealthCheckResult(
            passed=False,
            check_name="strategy_registry",
            message=f"strategy='{strategy_class}' NOT FOUND in registry and not a dotted path",
            severity="error",
            yaml_keys=["training.strategy_class"],
        )

    def check_loss_weight_declaration_ssot(
        self, config: TrainingSettings
    ) -> list[HealthCheckResult]:
        """One loss weight, one declaration surface — the domain list is the SSOT.

        A loss weight can be written two ways: on the *category* axis as
        ``losses.<section>.lambda_<name>``, or on the *domain* axis as an entry in
        ``losses.{image,kspace,complex,latent}_losses``. Since v6.0 the domain
        lists are the declarative form, and only a list entry actually builds the
        loss module and selects the FFT bridge (``output_domain`` x which list).
        A lambda written beside a list entry changes nothing, so the pair is safe
        to read and NOT symmetric to edit: moving the weight to the lambda
        silently drops the term, and editing one of the two leaves the arm
        declaring two different numbers for one objective.

        ``build_loss_weight_table`` already refuses a pair that *disagrees*. This
        check covers the pair that currently agrees — where the harm is not a
        wrong number but the absence of a single owner (non-negotiable 17).

        The pairing rule itself lives in :func:`dual_surface_loss_declarations`,
        which this check and the migration script share so that the script can
        never delete a field the audit would not have flagged.

        A name declared in **two lists** (``image_losses`` and ``kspace_losses``)
        also produces a joined ``source``, and that is a different question owned
        by ``check_loss_domain_consistency`` — so a pair is reported only when at
        least one side is a ``lambda_`` and at least one side is a list entry.
        One corpus arm has the list+list shape today
        (``promoted/exp_promoted_mri_slam.yaml``), and it is deliberately not
        flagged here.
        """
        results: list[HealthCheckResult] = []

        losses = getattr(config, "losses", None)
        if losses is None:
            return results

        try:
            duplicated = dual_surface_loss_declarations(losses)
        except Exception as exc:
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="loss_weight_declaration_ssot",
                    message=f"Loss weight table could not be built: {exc}",
                    severity="error",
                    yaml_keys=["losses"],
                    fix_hint=(
                        "Two surfaces declare the same loss at DIFFERENT weights. "
                        "Keep the domain-list entry and delete the lambda_ field."
                    ),
                )
            )
            return results

        if duplicated:
            lines = [
                f"  • '{name}' (weight={weight}) declared at {', '.join(lam)} AND {', '.join(lst)}"
                for name, weight, lam, lst in sorted(duplicated)
            ]
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="loss_weight_declaration_ssot",
                    message=(
                        f"{len(duplicated)} loss weight(s) are declared on BOTH the "
                        "lambda_ surface and a domain list. The domain list is the "
                        "SSOT since v6.0 — the lambda beside it builds nothing and "
                        "is free to drift:\n" + "\n".join(lines)
                    ),
                    severity="warning",
                    category="duplication",
                    yaml_keys=sorted(
                        {s.rsplit(".", 1)[0] for _, _, lam, _ in duplicated for s in lam}
                    ),
                    fix_hint=(
                        "Delete the lambda_ field and keep the domain-list entry. Do "
                        "NOT do the reverse: only a list entry builds the loss module "
                        "and selects the FFT bridge."
                    ),
                )
            )

        return results

    def check_physics_config(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Flag INERT physics on a k-space reconstruction/diffusion arm.

        The pre-#933 gate asked ``physics is None`` — unsatisfiable, since
        the schema always constructs a ``PhysicsConfigSchema`` (measured
        0/38 on a 216-arm sample), so the check could never fire. The
        answerable question is whether the physics block that DOES get
        constructed does anything: with data consistency disabled and no
        undersampling/sampling-mask block declared, a k-space forward model
        has nothing to condition the reconstruction on.

        Only k-space reconstruction/diffusion needs the physics block
        (sampling masks, coil maps, data-consistency operators). Image-
        domain super-resolution / denoising on NIfTI works fine without
        it — flagging those would be a noisy false positive.

        Blast radius (measured 2026-08-12 over 647 ``experiments/inprogress``
        arms, local CPU run — never a substitute for the cluster audit):
        120 arms are recon/diffusion-on-k-space (applicable), and 2 are
        physics-inert by this predicate — but at least one of the two is a
        DELIBERATE no-physics control, not a bug: ``workflow_baselines/
        b0_structural_denoise_m4raw.yaml`` documents in its own header "the
        only task on this cluster whose degradation is PHYSICALLY REAL... No
        acceleration, no physics block, no DC, no coil maps, no adapters"
        and declares ``workflow.task: denoising``. The check's premise —
        recon/diffusion on k-space needs DC or a sampling mask — does not
        hold for a documented denoising/restoration task, where the
        degradation being modelled is thermal noise, not missing k-space
        samples. Landing this at ``severity="error"`` would repeat the exact
        failure mode ``check_legacy_schema_mixing`` was DELETED for: firing
        on legitimate, documented, intentional use.
        ``task == "denoising"`` cannot be used as a blanket exemption either
        — the second flagged arm (``quality_matching/exp_qm_02b_restore_
        on_real.yaml``) is the same no-undersampling posture but declares
        ``workflow.task: reconstruction``, so a task-keyed allowlist would
        need to characterise more than one spelling and risks becoming its
        own guess. Given the brief's own conditional ("land as error ONLY IF
        each flagged arm genuinely has inert physics"), that condition is
        not met, so this lands ``severity="info"`` (advisory-first, the
        brief's other branch) rather than error. ``passed`` stays ``False``
        for the inert case — the finding itself (DC off, no undersampling
        block) is a definite structural fact, unlike the ``passed=True``
        "not yet measured" idiom used elsewhere in this file; only whether
        that fact is a BUG is undecided, and ``severity="info"`` is what
        keeps an undecided finding from blocking under ``--strict`` (see
        ``HealthCheckReport.passed`` / ``.warnings`` — both gate on
        ``severity``, not ``passed``, so ``info`` never blocks in either
        audit mode regardless of ``passed``). A task-aware exemption (or an
        explicit per-arm opt-out, mirroring
        ``synthetic_forward_probe_skip``) is the tracked ratchet-to-error
        follow-up once someone characterises the no-physics-task vocabulary
        properly, rather than guessing it under this task's scope.
        """
        results: list[HealthCheckResult] = []

        strategy_class = config.training.strategy_class if config.training else ""
        dataset_type = config.data.dataset_type if config.data is not None else None

        is_recon_or_diff = strategy_class is not None and any(
            m in strategy_class.lower() for m in ["diffusion", "reconstruction"]
        )
        operates_on_kspace = dataset_type == "kspace" or (
            config.data is not None
            and getattr(getattr(config.data, "coils", None), "processing_mode", None)
            in ("rss", "svd", "flatten")
            and not self._is_image_domain_dataset(dataset_type)
        )

        if not (is_recon_or_diff and operates_on_kspace):
            return results

        physics = config.physics
        dc_enabled = bool(getattr(getattr(physics, "data_consistency", None), "enabled", False))
        # "No sampling mask" is BLOCK ABSENCE, matching
        # ``check_acceleration_present``'s notion of the canonical
        # ``undersampling:`` block -- not ``undersampling.sampling_pattern``,
        # a rarely-set v6.1 alias (46/48 candidates that merely lacked the
        # alias DID declare an undersampling block, which would have made
        # the predicate over-broad).
        accel = getattr(config, "undersampling", None)
        physics_inert = not dc_enabled and accel is None

        if physics_inert:
            results.append(
                HealthCheckResult(
                    # advisory until a documented no-physics-task vocabulary
                    # exists to exempt legitimate denoising/restoration
                    # controls (see docstring) — 1+/2 flagged arms measured
                    # 2026-08-12 are deliberate, not bugs.
                    passed=False,
                    check_name="physics_config",
                    message=(
                        f"physics config is inert for k-space "
                        f"strategy='{strategy_class}' (dataset_type='{dataset_type}'): "
                        "physics.data_consistency.enabled is False and no "
                        "undersampling: block is declared. Nothing constrains "
                        "the reconstruction to the acquired measurements. "
                        "This may be intentional for a denoising/restoration "
                        "arm whose degradation is not k-space undersampling — "
                        "verify before treating this as a bug."
                    ),
                    severity="info",
                    category="physics_config",
                    yaml_keys=[
                        "physics.data_consistency.enabled",
                        "undersampling",
                    ],
                    fix_hint=(
                        "If this arm should be constrained by acquisition "
                        "physics: enable physics.data_consistency.enabled, "
                        "or declare an undersampling: block (e.g. "
                        "base_acceleration: 4). If this is a deliberate "
                        "no-physics denoising/restoration control (degradation "
                        "= noise, not undersampling), no action is needed."
                    ),
                )
            )
        else:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="physics_config",
                    message="physics config is live for this k-space arm.",
                    severity="info",
                    category="physics_config",
                )
            )

        return results

    # Models whose out_channels do not need to match in_channels
    # (e.g. evidential heads emit extra parameters per pixel).
    # TODO: replace with registry metadata once ModelRegistry exposes
    # per-entry channel-relation hints.
    _OUT_CHANNELS_INDEPENDENT_MODELS: frozenset[str] = frozenset(
        {
            "evidential_unet",
            # FNO reconstructs a (1- or 2-ch) image/k-space estimate whose
            # channel count is independent of the input coil count — an
            # svd(num_virtual_coils=4)→8-ch INPUT still maps to a 2-ch complex
            # reconstruction OUTPUT, so out_channels must not be checked
            # against the coil-derived expectation.
            "fno",
        }
    )

    # Models whose strategy concatenates an extra reference / conditioning
    # tensor onto the input at runtime, so model.in_channels does NOT match
    # the dataset's per-sample channel count. The audit can't see the
    # runtime concat, so it skips the in_channels check for these models
    # and emits an info-severity result instead of an error.
    #: Model types that *may* concatenate a runtime conditioning tensor onto
    #: the input. Membership is NECESSARY, not sufficient (#1387): for
    #: ``kspace_cold_diffusion`` the answer also depends on the backbone, so
    #: ask :meth:`_expects_input_concat` rather than testing this set directly.
    #: Testing it directly is what waived four channel checks on the six
    #: internal-DC arms, which concatenate nothing.
    _INPUT_CONCAT_MODELS: frozenset[str] = frozenset(
        {
            "disentangled_mri",  # guided_sr_strategy concats HF_reference
            "kspace_cold_diffusion",  # diffusion strategy MAY concat S-maps
        }
    )

    # Models that consume PAIRED-CONTRAST input — channels are doubled
    # because the M4Raw cross-contrast loader concatenates source + target
    # contrasts along the coil axis (single_contrast=false). The expected
    # channel count is therefore 2 × (single-contrast count). Used by
    # ``universal_multitask_dual`` (Pattern B fusion) so the
    # ``domain_alignment`` audit doesn't fire on the doubled in/out_channels.
    _PAIRED_CONTRAST_MODELS: frozenset[str] = frozenset(
        {
            "universal_multitask_dual",
            # ``kspace_cold_diffusion`` with ``dataset_type: m4raw`` and
            # ``single_contrast: false`` consumes [T1 || target] along the
            # coil axis (e.g. cross-contrast cold diffusion using T1 as a
            # fully-sampled prior), so per-sample channels are doubled.
            "kspace_cold_diffusion",
        }
    )

    def _is_paired_contrast(self, config: Any, model_type: str | None) -> bool:
        """Resolve whether this arm's loader actually fuses two contrasts.

        Membership in ``_PAIRED_CONTRAST_MODELS`` is NECESSARY but not
        SUFFICIENT for ``kspace_cold_diffusion``: the set's own comment already
        scopes that entry to ``dataset_type: m4raw`` with
        ``single_contrast: false``. That scope is real, not decorative —
        ``single_contrast`` is consumed by exactly one dataset
        (``M4RawRepetitionDataset``), which ``dataset_instantiator`` registers
        for ``dataset_type: m4raw`` alone, and nothing else in the data layer
        pairs contrasts. Testing membership directly is what re-exempted the
        six arms of #1387 at the two sites below: four declare
        ``dataset_type: kspace`` and two declare ``single_contrast: true``, so
        none of them pairs anything.

        ``universal_multitask_dual`` is deliberately left as bare membership —
        membership IS the answer for its Pattern B fusion, and it is outside
        #1387's scope. Do not fold it in without measuring its own arms.
        """
        if model_type not in self._PAIRED_CONTRAST_MODELS:
            return False
        if model_type != "kspace_cold_diffusion":
            return True

        # Bare attribute reads, NOT ``getattr(..., <default>)``. Both leaves are
        # declared fields on the schema (``data.dataset_type``,
        # ``data.pairing.single_contrast``), so on a real config a default is
        # unreachable — and a defaulting read is precisely the shape that
        # silently disabled ``rule_spatial_rank`` when its leaf moved into a
        # sub-block (see the same warning in
        # ``data/builders/dataset_instantiator.py``). Fail loud instead; a
        # fixture that cannot answer this is a fixture that does not model the
        # config the check runs on.
        return str(config.data.dataset_type) == "m4raw" and not config.data.pairing.single_contrast

    @staticmethod
    def _conditions_on_input(config: Any) -> bool:
        """``training.diffusion.condition_on_input`` on a non-cold, non-latent arm.

        Mirrors the gate in ``DiffusionTrainingStrategy._maybe_condition_on_input``:
        the concat is a no-op for cold and latent diffusion, so those arms keep the
        loaded width.
        """
        training = getattr(config, "training", None)
        diffusion = getattr(training, "diffusion", None)
        if diffusion is None or getattr(diffusion, "condition_on_input", None) is not True:
            return False
        # The strategy decides cold and latent from ``model.model_type``
        # (``KspaceMixin._is_cold_diffusion``, ``DiffusionTrainingStrategy
        # ._is_latent_diffusion``), not from ``training.diffusion``; read the
        # same source so the two cannot disagree on one arm.
        model_type = str(getattr(getattr(config, "model", None), "model_type", "") or "").lower()
        if "kspace_cold_diffusion" in model_type:
            return False
        if model_type in {"latent_diffusion", "latent_gaussian_diffusion", "ldm"}:
            return False
        return not ("latent" in model_type and "diffusion" in model_type)

    def _expects_input_concat(self, model: Any) -> bool:
        """Does this arm actually concatenate a conditioning tensor onto the input?

        Replaces the raw ``model_type in self._INPUT_CONCAT_MODELS`` test at
        every consumer (#1387). ``model`` is the ``config.model`` block, never a
        built module.

        ``model_type`` alone was never the invariant. For
        ``kspace_cold_diffusion`` the generator resolves S-map concatenation as
        ``condition_with_smaps AND backbone_type not in _INTERNAL_DC_BACKBONES``
        -- so the six ``diff_varnet`` / ``diff_varnet_kan`` arms are built at
        ``1x`` and concatenate nothing, yet the bare set test exempted them from
        the in_channels checks anyway. A declared/actual channel mismatch on
        those arms passed the audit and crashed at the first training batch.

        The conjunction is NOT restated here: it is imported from the generator,
        which owns it (CLAUDE.md #17). Re-listing the backbone names in this
        file would make the auditor a second resolver of the same rule -- the
        precise defect shape #1346 / #1453 had just been fixed.

        The import is function-local because it is the house style for
        ``spectramr.models`` reads in this module (ten existing sites) and keeps
        model construction out of the auditor's import graph.
        ``infrastructure -> models`` is an inward edge, so NN5 permits it.
        """
        model_type = getattr(model, "model_type", None)
        if model_type not in self._INPUT_CONCAT_MODELS:
            return False
        if model_type == "kspace_cold_diffusion":
            from spectramr.models.generators.kspace_cold_diffusion_generator import (
                config_expects_smaps_concat,
            )

            return config_expects_smaps_concat(getattr(model, "model_kwargs", None))
        # ``disentangled_mri``: guided_sr_strategy concats HF_reference
        # unconditionally, so membership IS the answer there.
        return True

    # Training strategies whose ``_forward_model`` UNCONDITIONALLY promotes the
    # image input to complex64 and real-stacks it ([B, C, H, W] complex →
    # [B, 2C, H, W] real) before the generator's nn.Conv2d. For a 1-channel
    # magnitude input (coil_processing_mode='rss_image' / 'magnitude' → 1
    # complex channel) this DOUBLES the model-visible channel count to 2, so
    # in_channels/out_channels must be 2, not 1. svd/flatten modes already
    # deliver real-stacked complex (2*N), so the round-trip is a no-op there —
    # the doubling below is guarded on ``expected_channels == 1`` to leave
    # those untouched. See virtual_fiducial_strategy._forward_model.
    #: Strategies that SYNTHESISE the model's input stack rather than passing
    #: the loaded batch through. For these, ``model.in_channels`` is decoupled
    #: from the coil pipeline entirely and the coil-channel arithmetic below
    #: describes the wrong quantity.
    _SYNTHESISED_STACK_STRATEGY_MARKERS: tuple[str, ...] = (
        "multi_acquisition_strategy",
        "ConcreteMultiAcquisitionStrategy",
    )

    _COMPLEX_STACKING_STRATEGY_MARKERS: tuple[str, ...] = (
        "virtual_fiducial_strategy",
        "motion_meta_strategy",
        "vf_admm_strategy",
        "ib_vf_strategy",
        # distillation_strategy._compute_losses_impl real-stacks the digital-twin
        # corrupted complex image (cat([corrupted.real, corrupted.imag])) AND the
        # clean target before the blind-student generator, so a 1-channel
        # rss_image magnitude becomes 2 model-visible channels. Missing here, the
        # audit expected in=1 and rejected the correct in=2 for
        # eval_c2/eval_c3/eval_c7/exp_c4 (cluster smoke 20260605, job 7095209).
        "distillation_strategy",
    )

    # dataset_type values whose loaders materialise the image-domain key
    # (not the `kspace` key). _apply_coil_processing's RSS / SVD branches
    # then take the is_kspace=False path, which collapses to 1 channel.
    #
    # Every dataset here whose signal domain is {"image"} in the SSOT
    # (data.datasets.axis_exposure.DATASET_TYPE_SIGNAL_DOMAINS) must also appear
    # there, or the two tables disagree about the same dataset. `pde_synthetic` /
    # `synthetic` are image-KEY-loading but not MR signals, so they are here but
    # NOT in the signal-domain SSOT. The overlap is pinned by
    # tests/unit/infrastructure/validation/test_workflow_dataset_signal_domain.py.
    _IMAGE_DOMAIN_DATASET_TYPES: frozenset[str] = frozenset(
        {
            "image",
            "nifti",
            "nifti_paired",
            "dicom",
            "contrast_aware_paired",
            # MRIxFields2026 is magnitude-only [0, 1] MNI images. It materialises the
            # image key, so RSS/SVD must take the 1-channel path — omitting it made
            # _derive_expected_channels return 2 (real+imag) for magnitude data,
            # mis-aligning the channel check for every mrixfields arm under rss.
            "mrixfields",
            # `pde_synthetic` produces 1-channel real-valued spatial
            # tensors directly (no coils, no Fourier domain) — treat as
            # image-domain so the channel-alignment check stays correct.
            "pde_synthetic",
            # Smoke-test synthetic also lives in image space.
            "synthetic",
        }
    )

    @staticmethod
    def _is_image_domain_dataset(dataset_type: str | None) -> bool:
        """Return True if `dataset_type` loads image-space tensors."""
        if not dataset_type:
            return False
        return dataset_type in ConfigHealthChecker._IMAGE_DOMAIN_DATASET_TYPES

    def check_marker_kappa_in_tissue_range(self, config: TrainingSettings) -> HealthCheckResult:
        """The fiducial must calibrate INSIDE the range it is calibrating.

        A single-material marker pins ONE point on the contrast-transfer curve;
        every other tissue's kappa is the Bottomley model extrapolated, not
        measured. That is a real limit either way, but it becomes a wrong claim
        when the marker's own kappa falls OUTSIDE the tissue range, because the
        one measured point is then not even bracketing what it certifies.

        Measured over 64 mT -> 3 T at TR 500 / 90 degrees the tissue range is
        [0.453, 0.865]; a short-T1 marker (50 -> 60 ms) gives 0.9998, outside it
        entirely, and nothing rejected that before 2026-07-26.

        Lives here rather than on the schema because computing kappa needs
        ``infrastructure.physics``, and ``config/`` may not import leftward
        (CLAUDE.md #5).
        """
        macq = getattr(getattr(config, "physics", None), "multi_acquisition", None)
        relax = getattr(macq, "relaxometric_calibration", None)
        if relax is None or not getattr(relax, "enabled", False):
            return HealthCheckResult(
                passed=True,
                check_name="marker_kappa_in_tissue_range",
                message="relaxometric_calibration disabled — nothing to calibrate.",
                severity="info",
            )

        from spectramr.infrastructure.physics.relaxometric_calibration import (
            AcquisitionParams,
            relaxometric_gain,
            tissue_gain_map,
        )

        src = AcquisitionParams(
            relax.source.field_strength_t,
            relax.source.tr_ms,
            relax.source.te_ms,
            relax.source.flip_deg,
        )
        tgt = AcquisitionParams(
            relax.target.field_strength_t,
            relax.target.tr_ms,
            relax.target.te_ms,
            relax.target.flip_deg,
        )
        gains = tissue_gain_map(src, tgt)
        lo, hi = min(gains.values()), max(gains.values())
        marker = relaxometric_gain(
            relax.marker_t1_ms,
            relax.marker_t1_target_ms or relax.marker_t1_ms,
            relax.marker_t2_ms,
            src,
            tgt,
        )
        inside = lo <= marker <= hi
        return HealthCheckResult(
            passed=inside,
            check_name="marker_kappa_in_tissue_range",
            message=(
                f"marker kappa={marker:.4f} from T1 {relax.marker_t1_ms:.0f}->"
                f"{relax.marker_t1_target_ms or relax.marker_t1_ms:.0f} ms, T2 "
                f"{relax.marker_t2_ms:.0f} ms; tissue range "
                f"[{lo:.4f}, {hi:.4f}] ({hi / lo:.2f}x spread)."
                + (
                    " The marker pins ONE point inside that range; the rest is "
                    "the Bottomley model extrapolated, not measured. A "
                    "multi-compartment marker would pin several."
                    if inside
                    else " OUTSIDE the range: the marker calibrates by "
                    "extrapolation, so a kappa verified on it says nothing about "
                    "any tissue the network actually has to translate. Declare "
                    "marker relaxometry that brackets the tissues."
                )
            ),
            severity="info" if inside else "error",
        )

    def check_domain_alignment(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Validate model channel counts against data pipeline output.

        This pre-flight check catches DomainMismatch errors BEFORE the
        expensive model + data pipeline are instantiated, by computing
        the expected channel count from the data configuration and
        comparing it to model.in_channels / model.out_channels.

        Channel mapping rules (mirrored from
        ``torchio_subject_builder._apply_coil_processing``):

        | coil_processing_mode | dataset_type | Expected channels          |
        |----------------------|--------------|----------------------------|
        | rss                  | kspace       | 2 (real + imaginary)       |
        | rss                  | image*       | 1 (real magnitude)         |
        | rss_image            | any          | 1 (RSS magnitude image)    |
        | magnitude            | any          | 1 (real magnitude)         |
        | svd                  | kspace       | 2 * num_virtual_coils      |
        | svd                  | image*       | 1 (falls back to RSS mag.) |
        | flatten              | kspace       | 2 * num_physical_coils     |
        | none                 | kspace       | depends on file (skipped)  |
        | (any)                | synthetic    | depends on generator       |

        * "image" includes `image`, `nifti`, `nifti_paired`, `dicom`,
          `contrast_aware_paired` — every dataset_type whose loader
          materialises the *image-domain* key (not the `kspace` key).
          ``_apply_coil_processing`` switches on that runtime flag, so
          the audit must mirror it from the config to stay correct.
        """
        results: list[HealthCheckResult] = []

        if not config.data or not config.model:
            return results

        coil_mode = config.data.coils.processing_mode
        dataset_type = config.data.dataset_type
        in_channels = config.model.in_channels
        out_channels = config.model.out_channels
        target_channels = config.data.domain.target_channels
        model_type = config.model.model_type or ""

        if in_channels is None:
            return results  # Cannot validate without explicit in_channels

        # 1-D sequence / navigator encoders (e.g. ``nav_encoder_1d`` in the
        # IB-VF arm) consume a 1-D signal the *strategy* extracts from the 2-D
        # acquisition (the DC k-space navigator m(t)=y(0,t)), NOT the coil
        # image — so in_channels is the navigator's channel count (2 = Re/Im)
        # and out_channels its latent dim, neither tied to the coil count. The
        # coil-channel math below does not apply. (smoke_audit_20260526 / F9)
        if getattr(config.model, "spatial_dims", None) == 1:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="domain_alignment",
                    message=(
                        f"model.spatial_dims=1 → 1-D sequence/navigator encoder "
                        f"'{model_type}'; the training strategy extracts a 1-D "
                        f"signal from the 2-D acquisition, so coil-channel "
                        f"alignment is not applicable (in_channels={in_channels}, "
                        f"out_channels={out_channels})."
                    ),
                    severity="info",
                )
            )
            return results

        # Compute expected channel count from data pipeline config.
        # `expected_channels is None` means the count cannot be derived
        # statically (synthetic data, raw passthrough, file-dependent
        # coil count) — emit an info-severity result and skip the
        # match check rather than guess.
        expected_channels, reason = self._derive_expected_channels(
            coil_mode=coil_mode,
            dataset_type=dataset_type,
            num_virtual_coils=config.data.coils.num_virtual_coils,
            in_channels=in_channels,
        )

        # Paired-contrast fusion models consume both source + target
        # contrasts in one tensor (M4Raw cross-contrast loader cats them
        # along the coil axis when ``single_contrast=false``), so the
        # expected per-sample channel count is doubled.
        paired_contrast_doubled = False
        single_contrast_channels = expected_channels
        if (
            expected_channels is not None
            and self._is_paired_contrast(config, model_type)
            # Retained as the ``universal_multitask_dual`` clause: the predicate
            # folds ``single_contrast`` in for ``kspace_cold_diffusion`` only, so
            # u_m_d keeps exactly the condition it had before #1387.
            and not config.data.pairing.single_contrast
        ):
            paired_contrast_doubled = True
            expected_channels = expected_channels * 2
            reason = (
                f"{reason}; doubled to {expected_channels} for "
                f"paired-contrast model '{model_type}' "
                f"(single_contrast=false → source+target concatenated)"
            )

        # A conditional diffusion arm concatenates the measurement onto the noised
        # target (``training.diffusion.condition_on_input: true``,
        # ``DiffusionTrainingStrategy._maybe_condition_on_input``; a no-op for cold
        # and latent diffusion), so the model's INPUT is twice the loaded width
        # while its output stays the target width. hilbert_mamba/exp_hm_09 declared
        # in_channels=2 for a 1-channel loader and was rejected (2026-09-03).
        input_concat_doubled = False
        if expected_channels is not None and self._conditions_on_input(config):
            input_concat_doubled = True
            expected_channels = expected_channels * 2
            reason = (
                f"{reason}; doubled to {expected_channels} for the input because "
                "training.diffusion.condition_on_input=true concatenates the "
                "measurement onto the noised target (output stays "
                f"{single_contrast_channels})"
            )

        # Multi-acquisition strategies never feed the loaded batch to the
        # generator: they SYNTHESISE an acquisition stack whose width comes from
        # ``physics.multi_acquisition``, so the coil arithmetic above describes a
        # tensor the model never sees. Before 2026-07-26 these arms happened to
        # sit on ``coil_processing_mode: none`` + ``dataset_type: kspace``, where
        # the count is file-dependent and the check skipped; moving them to an
        # image dataset made it derive 1 and reject a correct in_channels of 24.
        #
        # Checking the RIGHT quantity beats skipping: for ``subvoxel_sr`` the
        # contract is n_frames frames + 2n shift-conditioning maps, plus n more
        # when the fiducial is fed to the model — the same arithmetic the
        # strategy raises on at setup, now caught at audit time instead.
        _training_cfg = getattr(config, "training", None)
        _strategy = getattr(_training_cfg, "strategy_class", None) or ""
        macq = getattr(getattr(config, "physics", None), "multi_acquisition", None)
        if (
            isinstance(_strategy, str)
            and any(m in _strategy for m in self._SYNTHESISED_STACK_STRATEGY_MARKERS)
            and macq is not None
            and getattr(macq, "enabled", False)
        ):
            if getattr(macq, "method", None) == "subvoxel_sr":
                per_frame = 4 if getattr(macq, "marker_channels", False) else 3
                expected = int(macq.n_frames) * per_frame
                passed = in_channels == expected
                results.append(
                    HealthCheckResult(
                        passed=passed,
                        check_name="domain_alignment",
                        message=(
                            f"subvoxel_sr synthesises {macq.n_frames} frames + "
                            f"{2 * macq.n_frames} shift maps"
                            + (f" + {macq.n_frames} fiducial frames" if per_frame == 4 else "")
                            + f" = {expected} channels; model.in_channels="
                            f"{in_channels}. The coil pipeline does not set this "
                            "width — the strategy does."
                        ),
                        severity="info" if passed else "error",
                    )
                )
                return results
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="domain_alignment",
                    message=(
                        f"training strategy '{_strategy.rsplit('.', 1)[-1]}' "
                        f"synthesises the model input for method "
                        f"'{getattr(macq, 'method', None)}', so in_channels="
                        f"{in_channels} is decoupled from the coil pipeline; "
                        "skipping the coil-channel comparison."
                    ),
                    severity="info",
                )
            )
            return results

        # VF-family strategies real-stack a 1-channel complex magnitude into 2
        # real channels before the generator (see
        # virtual_fiducial_strategy._forward_model). A naive count would expect
        # 1 (rss_image) and reject the correct in_channels=2 at audit time —
        # turning a runtime DomainMismatch crash into an audit skip. Guarded on
        # ``expected_channels == 1`` so svd VF arms (already 2*N real) are
        # unaffected.
        training_cfg = getattr(config, "training", None)
        strategy_class = getattr(training_cfg, "strategy_class", None) or ""
        if (
            expected_channels == 1
            and isinstance(strategy_class, str)
            and any(marker in strategy_class for marker in self._COMPLEX_STACKING_STRATEGY_MARKERS)
        ):
            expected_channels = 2
            reason = (
                f"{reason}; doubled to 2 because the training strategy "
                f"'{strategy_class.rsplit('.', 1)[-1]}' real-stacks the complex "
                f"magnitude (1 complex → 2 real channels) before the generator"
            )

        if expected_channels is None:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="domain_alignment",
                    message=(f"{reason} (in_channels={in_channels}, out_channels={out_channels})"),
                    severity="info",
                )
            )
            # Run the cheap parity check for `flatten` which can be done
            # without knowing the absolute coil count.
            if coil_mode == "flatten" and in_channels % 2 != 0:
                results.append(
                    HealthCheckResult(
                        passed=False,
                        check_name="domain_alignment",
                        message=(
                            f"coil_processing_mode='flatten' produces 2*N_coils "
                            f"channels, but in_channels={in_channels} is odd"
                        ),
                        severity="error",
                    )
                )
            # `none` + kspace must still have at least 2 channels for complex data.
            # All canonical kspace dataset_type variants must be matched —
            # see ``src/config/schemas/data.py`` for the alias table
            # (kspace, fastmri_kspace, kspace_fastmri, fastmri_knee, m4raw
            # all normalise to k-space input). Matching only the literal
            # ``"kspace"`` would silently let the more common
            # ``"fastmri_kspace"`` slip past this safety check.
            _KSPACE_TYPES = {
                "kspace",
                "fastmri_kspace",
                "kspace_fastmri",
                "fastmri_knee",
                "m4raw",
            }
            if coil_mode == "none" and dataset_type in _KSPACE_TYPES and in_channels < 2:
                results.append(
                    HealthCheckResult(
                        passed=False,
                        check_name="domain_alignment",
                        message=(
                            f"coil_processing_mode='none' with "
                            f"dataset_type='{dataset_type}' passes raw multi-coil "
                            f"data through, but in_channels={in_channels} is too "
                            f"small (minimum 2)"
                        ),
                        severity="error",
                    )
                )
            return results

        # F-DOMAIN-ADAPTER-FOLD / 2026-05-20 — if the YAML declares a
        # ``pre_model`` adapter chain composed entirely of adapters
        # whose channel transforms are known to ``_ADAPTER_CHANNEL_EFFECTS``,
        # we can fold the chain and compare the FOLDED output to
        # ``in_channels``. Without this, the F-FNO-OUTCHAN YAML (which
        # bridges 1-ch image → 2-ch interleaved-complex via
        # ``fft_image_to_kspace + complex_to_real_imag_interleave``)
        # would fire a domain_alignment error even though the chain
        # resolves cleanly (caught by ``check_adapter_chain_channel_resolution``).
        adapter_resolved_channels: int | None = None
        adapters_block = getattr(config, "adapters", None)
        pre_model_chain = (
            list(getattr(adapters_block, "pre_model", []) or [])
            if adapters_block is not None
            else []
        )
        if pre_model_chain and expected_channels is not None:
            chain_names = [getattr(step, "name", None) for step in pre_model_chain]
            chain_names = [n for n in chain_names if n]
            if chain_names and all(n in self._ADAPTER_CHANNEL_EFFECTS for n in chain_names):
                folded = expected_channels
                for n in chain_names:
                    folded = self._ADAPTER_CHANNEL_EFFECTS[n](folded)
                adapter_resolved_channels = folded

        # Validate in_channels against expected. Skip the strict check for
        # models whose strategy concatenates a runtime conditioning tensor
        # onto the input — the audit can't see that concat, so a literal
        # equality check would always fire.
        if self._expects_input_concat(getattr(config, "model", None)):
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="domain_alignment",
                    message=(
                        f"model.in_channels={in_channels} not strictly checked: "
                        f"model_type='{model_type}' concatenates a runtime "
                        f"conditioning tensor (reference / sensitivity maps) "
                        f"onto the input, which the static audit cannot see. "
                        f"Resolved per-arm from model_kwargs, not from "
                        f"model_type alone (#1387)."
                    ),
                    severity="info",
                )
            )
        elif adapter_resolved_channels is not None:
            # Trust the adapter-folded count.
            in_ok = in_channels == adapter_resolved_channels
            results.append(
                HealthCheckResult(
                    passed=in_ok,
                    check_name="domain_alignment",
                    message=(
                        f"model.in_channels={in_channels} "
                        f"{'matches' if in_ok else 'MISMATCHES'} "
                        f"adapter-folded expected={adapter_resolved_channels} "
                        f"(data={expected_channels} → pre_model chain → "
                        f"{adapter_resolved_channels})"
                    ),
                    severity="error" if not in_ok else "info",
                )
            )
            # Also relax the out_channels check: the user's declared
            # ``out_channels`` is the model's emitted width, which the
            # ``pre_loss_pred`` chain will bridge back to expected for
            # the loss. Skip the strict comparison in this branch.
            return results
        else:
            in_ok = in_channels == expected_channels
            results.append(
                HealthCheckResult(
                    passed=in_ok,
                    check_name="domain_alignment",
                    message=(
                        f"model.in_channels={in_channels} "
                        f"{'matches' if in_ok else 'MISMATCHES'} "
                        f"expected={expected_channels} ({reason})"
                    ),
                    severity="error" if not in_ok else "info",
                )
            )

        # Validate out_channels (typically matches in_channels for reconstruction)
        if out_channels is not None:
            if model_type in self._OUT_CHANNELS_INDEPENDENT_MODELS:
                results.append(
                    HealthCheckResult(
                        passed=True,
                        check_name="domain_alignment",
                        message=(
                            f"model.out_channels={out_channels} not checked: "
                            f"model_type='{model_type}' declares its own out-channel "
                            f"semantics"
                        ),
                        severity="info",
                    )
                )
            else:
                out_ok = out_channels == expected_channels
                # Paired-contrast: a model may emit just the *target*
                # contrast (single-contrast count) instead of the full
                # paired tensor — both are legitimate. Accept either.
                if (
                    not out_ok
                    and (paired_contrast_doubled or input_concat_doubled)
                    and out_channels == single_contrast_channels
                ):
                    out_ok = True
                results.append(
                    HealthCheckResult(
                        passed=out_ok,
                        check_name="domain_alignment",
                        message=(
                            f"model.out_channels={out_channels} "
                            f"{'matches' if out_ok else 'MISMATCHES'} "
                            f"expected={expected_channels} ({reason})"
                        ),
                        severity="error" if not out_ok else "info",
                    )
                )

        # Validate target_channels if present
        if target_channels is not None:
            tgt_ok = target_channels == expected_channels
            # ``target_channels: 1`` is the standard fastMRI RSS-magnitude
            # reference convention: regardless of how the model emits its
            # prediction (k-space or image, multi-coil or virtual coils),
            # the validation/loss target is a 1-ch magnitude image
            # produced by RSS-combining the multi-coil ground truth.
            # Strategies (kspace_cold_diffusion, varnet, etc.) handle
            # the model-output → 1-ch RSS reduction internally before
            # the loss compare. Accepting this for any coil-encoding
            # mode (rss, svd, flatten) treats it as the convention it
            # actually is, not a misconfiguration.
            if (
                not tgt_ok
                and coil_mode in ("rss", "rss_image", "magnitude", "svd", "flatten")
                and target_channels == 1
            ):
                tgt_ok = True
            # Paired-contrast: ``target_channels`` typically points at the
            # target-contrast slice of the [source||target] tensor (e.g.
            # cross-contrast cold diffusion with target_channels=8 in a
            # 16-channel tensor → loss on target contrast only).
            if (
                not tgt_ok
                and paired_contrast_doubled
                and target_channels == single_contrast_channels
            ):
                tgt_ok = True
            results.append(
                HealthCheckResult(
                    passed=tgt_ok,
                    check_name="domain_alignment",
                    message=(
                        f"data.domain.target_channels={target_channels} "
                        f"{'matches' if tgt_ok else 'MISMATCHES'} "
                        f"expected={expected_channels} ({reason})"
                    ),
                    severity="warning" if not tgt_ok else "info",
                )
            )

        return results

    @staticmethod
    def _derive_expected_channels(
        coil_mode: str,
        dataset_type: str,
        num_virtual_coils: int | None,
        in_channels: int,
    ) -> tuple[int | None, str]:
        """Compute the expected channel count from data-pipeline config.

        Returns (expected, reason). `expected is None` means the count
        cannot be derived without inspecting actual data — caller should
        emit an info-severity result rather than guess.
        """
        if dataset_type == "synthetic":
            return None, "dataset_type='synthetic' — skipping strict channel check"

        if coil_mode == "rss":
            if ConfigHealthChecker._is_image_domain_dataset(dataset_type):
                return (
                    1,
                    f"coil_processing_mode='rss' on image-domain "
                    f"dataset_type='{dataset_type}' collapses to "
                    f"1-channel magnitude (mirrors "
                    f"_apply_coil_processing's is_kspace=False branch)",
                )
            return 2, "coil_processing_mode='rss' produces 2 channels (real+imag RSS)"

        if coil_mode == "magnitude":
            return 1, "coil_processing_mode='magnitude' produces 1 channel (magnitude)"

        if coil_mode == "rss_image":
            return (
                1,
                "coil_processing_mode='rss_image' produces 1 channel (RSS magnitude image; k-space inputs IFFT-ed first)",
            )

        if coil_mode == "svd":
            if ConfigHealthChecker._is_image_domain_dataset(dataset_type):
                # _apply_coil_processing's image-domain SVD branch logs a
                # warning and falls back to RSS magnitude (1 channel).
                return (
                    1,
                    f"coil_processing_mode='svd' on image-domain "
                    f"dataset_type='{dataset_type}' falls back to "
                    f"1-channel RSS magnitude",
                )
            if num_virtual_coils is None:
                return (
                    None,
                    "coil_processing_mode='svd' but num_virtual_coils is unset — "
                    "skipping strict channel check",
                )
            expected = 2 * num_virtual_coils
            return (
                expected,
                f"coil_processing_mode='svd' with num_virtual_coils={num_virtual_coils} "
                f"produces {expected} channels (2 × {num_virtual_coils})",
            )

        if coil_mode == "flatten":
            return (
                None,
                f"coil_processing_mode='flatten' with in_channels={in_channels} "
                f"(implies {in_channels // 2} physical coils) — exact count "
                f"depends on file",
            )

        if coil_mode == "none":
            return (
                None,
                f"coil_processing_mode='none' — data channels depend on file. "
                f"Model expects in_channels={in_channels}",
            )

        return (
            None,
            f"coil_processing_mode='{coil_mode}' is unrecognized — skipping check",
        )

    # ------------------------------------------------------------------
    # Tier-1 audit-ladder additions (2026-05-03 spec):
    # - check_advertised_options:    forbid silent model fallbacks
    # - check_loss_domain_consistency: image losses vs output_domain
    # - check_amp_grad_clip_interaction: AMP + manual unscale_ trap
    # ------------------------------------------------------------------

    def check_advertised_options(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Reject ``model_kwargs`` that are not in a model's advertised set.

        A model may declare an ``OPTION_SCHEMA`` class attribute mapping
        kwarg names to allowed values, e.g.::

            class MyModel(nn.Module):
                OPTION_SCHEMA = {"attention_type": ["spatial", "cross"]}

        If the YAML sets ``model.model_kwargs.attention_type='dual_domain'``
        and the model's ``OPTION_SCHEMA["attention_type"]`` does not
        include it, this check raises a hard error rather than letting
        the model silently fall back to ``"spatial"`` at runtime.

        Models without an ``OPTION_SCHEMA`` attribute are skipped
        (backward compatible).
        """
        results: list[HealthCheckResult] = []
        model_cfg = getattr(config, "model", None)
        if model_cfg is None:
            return results
        model_type = getattr(model_cfg, "model_type", None)
        kwargs = getattr(model_cfg, "model_kwargs", None) or {}
        if model_type is None or not kwargs:
            return results

        try:
            from spectramr.models.registry import get_model_class  # lazy

            cls = get_model_class(model_type)
        except Exception:
            return results  # registry missing — caught by check_model_registry

        schema = getattr(cls, "OPTION_SCHEMA", None) or {}

        # ── Signature advisory (pitfall #16, 2026-06-12) ────────────────
        # For kwargs NOT covered by an OPTION_SCHEMA entry: a key that is
        # neither an explicit ``__init__`` parameter nor absorbable by
        # ``**kwargs`` is certainly either swallowed by the factory's
        # signature filter (a facade knob — the run silently ignores it) or
        # a ``TypeError`` at build time. Surfaced at *info* severity — an
        # advisory, never blocking, until a cluster-wide audit confirms the
        # existing-arm blast radius is clean.
        try:
            import inspect

            sig_params = inspect.signature(cls.__init__).parameters
            has_var_kwargs = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
            )
        except (TypeError, ValueError):  # C-extension / exotic __init__
            sig_params, has_var_kwargs = {}, True

        if not has_var_kwargs:
            for key in kwargs:
                if key in schema or key in sig_params:
                    continue
                results.append(
                    HealthCheckResult(
                        passed=False,
                        check_name="advertised_options",
                        message=(
                            f"model.model_kwargs.{key} is not an "
                            f"__init__ parameter of {model_type!r} (and the "
                            "signature has no **kwargs): the key is either "
                            "silently dropped by the factory's signature "
                            "filter (pitfall #16 facade knob) or a TypeError "
                            "at build."
                        ),
                        severity="info",
                        category="advertised_options",
                        yaml_keys=[f"model.model_kwargs.{key}"],
                        fix_hint=(
                            f"Remove model.model_kwargs.{key}, rename it to a "
                            f"real {model_type} parameter, or declare it in "
                            f"{model_type}.OPTION_SCHEMA."
                        ),
                    )
                )

        for key, allowed in schema.items():
            if key not in kwargs:
                continue
            value = kwargs[key]
            if value not in allowed:
                results.append(
                    HealthCheckResult(
                        passed=False,
                        check_name="advertised_options",
                        message=(
                            f"model.model_kwargs.{key}={value!r} is not in the "
                            f"advertised set for {model_type!r}: {sorted(allowed)}. "
                            "Silent fallbacks are forbidden."
                        ),
                        severity="error",
                        category="advertised_options",
                        yaml_keys=[f"model.model_kwargs.{key}"],
                        fix_hint=(
                            f"Set model.model_kwargs.{key} to one of "
                            f"{sorted(allowed)} OR add {value!r} to "
                            f"{model_type}.OPTION_SCHEMA[{key!r}]."
                        ),
                    )
                )
        # Only claim a pass when an advertised set was actually consulted.
        # A model with no OPTION_SCHEMA (or one whose schema covers none of the
        # declared kwargs) is SKIPPED, exactly as this method's docstring says.
        # Emitting "All model_kwargs are in the advertised set" for a model whose
        # advertised set was never read asserts a fact the check did not
        # establish -- a confident green with nothing behind it, which is the
        # pitfall-#16 shape this checker exists to catch. `audit` is --strict by
        # default, so an unfounded `info` is not harmless: it is the reassuring
        # noise that hides the real finding. Issue #284.
        checked = sorted(key for key in schema if key in kwargs)
        if checked and not results:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="advertised_options",
                    message=(
                        f"All checked {model_type} model_kwargs are in the "
                        f"advertised set: {checked}."
                    ),
                    severity="info",
                    category="advertised_options",
                )
            )
        return results

    def check_loss_domain_consistency(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Flag image-domain losses declared in a kspace-output config (and vice-versa).

        v6.0 declarative loss schema groups losses into ``image_losses``,
        ``kspace_losses``, ``complex_losses``. The combination must
        agree with ``losses.output_domain``. Mixing them silently breaks
        the validation metrics (the FID-on-multi-coil failure in the
        last smoke run is the prototype).
        """
        results: list[HealthCheckResult] = []
        losses_cfg = getattr(config, "losses", None)
        if losses_cfg is None:
            return results
        output_domain = losses_cfg.policy.output_domain if losses_cfg else None
        if not output_domain:
            return results

        # NB: the canonical output_domain value for complex output is
        # ``complex_image`` — that is what LossBuilder._build_list_based_losses
        # honours (it raises on any other value). The old ``complex`` key was
        # dead (no config could legally carry it) and meant a phase-aware arm
        # declaring ``output_domain: complex_image`` + ``complex_losses`` was
        # falsely flagged as a domain mismatch (smoke 2026-05-25: twin_dps,
        # ib_infonce). Key on the value the builder accepts.
        compat = {
            "image": ("image_losses",),
            "kspace": ("kspace_losses",),
            "complex_image": ("complex_losses",),
        }
        # An explicit `adapters.pre_loss_pred` chain bridges any
        # output_domain → loss-domain mismatch the user has acknowledged.
        # Honor it here so the audit doesn't double-flag what the
        # adapters layer already covers.
        adapters_cfg = getattr(config, "adapters", None)
        explicit_pre_loss = bool(
            adapters_cfg is not None and (getattr(adapters_cfg, "pre_loss_pred", None) or [])
        )

        # Anything in a non-matching list is flagged. Severity depends on
        # direction:
        # - image_losses on a kspace-output model is an ERROR (the path
        #   needs to know how to invert k-space to image and that's fragile).
        # - kspace_losses on an image-output model is INFO: the schema
        #   docstring at LossConfigSchema.kspace_losses explicitly
        #   documents that an FFT bridge is inserted automatically, so
        #   this is the supported pattern, not a smell.
        # - complex_losses on an image-output model is INFO for the same
        #   reason (complex_losses docstring documents the cast).
        # Per LossBuilder._build_list_based_losses bridge matrix
        # (loss_builder.py:429-514) — mirror it EXACTLY so the audit never
        # contradicts what the builder actually does:
        #   output_domain=kspace:        image→iFFT-mag, complex→iFFT-complex
        #   output_domain=image:         complex→cast-to-complex (native-ish);
        #                                kspace is the ONLY fragile combo
        #                                (must self-FFT or declare an explicit
        #                                adapters.pre_loss_pred chain)
        #   output_domain=complex_image: image→magnitude-extract, kspace→FFT
        # The only genuinely-unbridged combo left flagged as error is
        # (image, kspace_losses), matching the builder's own warning.
        bridged_combos = {
            ("kspace", "image_losses"),
            ("kspace", "complex_losses"),
            ("image", "complex_losses"),
            ("complex_image", "image_losses"),
            ("complex_image", "kspace_losses"),
        }
        # DELIBERATELY NOT LOSS_LIST_DOMAINS (#1924). Every other "walk the
        # declared lists" loop in this file is derived from that SSOT; this one is
        # not, because it is not a walk -- it is the mirror of LossBuilder's bridge
        # matrix above, and that matrix defines NO bridge into a latent: the encoder
        # that would produce one is a learned map, not a transform (see
        # LossConfigSchema.latent_losses). ``latent_losses`` is legal only with
        # ``output_domain: latent``, a combination ``compat`` does not list, so
        # adding it here would flag every correct latent arm as a domain mismatch.
        # The schema validator owns that pairing instead. If a latent bridge is ever
        # defined, add it to ``compat``/``bridged_combos`` FIRST, then to this loop.
        for attr_name in ("image_losses", "kspace_losses", "complex_losses"):
            entries = getattr(losses_cfg, attr_name, None) or []
            # F-LOSSDOMAIN-ENABLED / 2026-05-20 — only consider losses
            # whose ``enabled`` flag is True (or absent / non-explicit).
            # An ``enabled: false`` entry documents an off-by-default
            # auxiliary loss and should NOT trip the loss-domain check
            # — it never fires at runtime, so it cannot be "silently
            # ignored or break the metric path". Previously the check
            # looked at list membership alone, forcing the user to
            # delete-rather-than-comment any cross-domain placeholder.
            entries = [e for e in entries if getattr(e, "enabled", True)]
            if not entries:
                continue
            domain_of_attr = attr_name.replace("_losses", "")
            if attr_name not in compat.get(output_domain, ()):
                names = [getattr(e, "name", "<unknown>") for e in entries]
                auto_bridged = (output_domain, attr_name) in bridged_combos
                # Explicit adapter chain on pre_loss_pred subsumes the
                # auto-bridge — even directions LossBuilder doesn't
                # natively bridge are valid when the user declares.
                bridged = auto_bridged or explicit_pre_loss
                severity = "info" if bridged else "error"
                if auto_bridged:
                    message = (
                        f"losses.output_domain={output_domain!r} with "
                        f"losses.{attr_name} ({names}): bridge is auto-inserted "
                        "by LossBuilder per schema contract."
                    )
                    fix_hint = None
                elif explicit_pre_loss:
                    message = (
                        f"losses.output_domain={output_domain!r} with "
                        f"losses.{attr_name} ({names}): bridged by explicit "
                        "adapters.pre_loss_pred chain."
                    )
                    fix_hint = None
                else:
                    message = (
                        f"losses.output_domain={output_domain!r} but "
                        f"losses.{attr_name} is non-empty ({names}). "
                        f"Losses in {attr_name} produce {domain_of_attr}-domain "
                        "signals — they will be silently ignored or break the "
                        "metric path."
                    )
                    fix_hint = (
                        f"Move the {names} entries into "
                        f"losses.{output_domain}_losses, OR set "
                        f"losses.output_domain to {domain_of_attr!r}."
                    )
                results.append(
                    HealthCheckResult(
                        passed=bridged,
                        check_name="loss_domain_consistency",
                        message=message,
                        severity=severity,
                        category="loss_domain_consistency",
                        yaml_keys=[f"losses.{attr_name}", "losses.output_domain"],
                        fix_hint=fix_hint,
                    )
                )

        if not results:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="loss_domain_consistency",
                    message=f"All loss declarations match output_domain={output_domain!r}.",
                    severity="info",
                    category="loss_domain_consistency",
                )
            )
        return results

    def check_amp_grad_clip_interaction(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Warn when AMP + manual unscale_ + gradient clipping race conditions are likely.

        The 6 ``unscale_() called twice`` failures from the last smoke
        come from the AMP wrapper unscaling once for clipping and once
        for the optimizer step without an intermediate ``update()``.
        We can't fully prove the race statically, but we can flag the
        risk-prone combination and let Tier 2 (synthetic forward probe)
        confirm.
        """
        results: list[HealthCheckResult] = []
        opt_cfg = getattr(config, "optimization", None)
        if opt_cfg is None:
            return results
        use_amp = bool(getattr(opt_cfg.precision, "enabled", False))
        clip = bool(getattr(opt_cfg.gradient.clip, "enabled", False))
        clip_value = getattr(opt_cfg.gradient.clip, "value", None)
        if not use_amp:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="amp_grad_clip_interaction",
                    message="AMP disabled; double-unscale risk is not applicable.",
                    severity="info",
                    category="amp_grad_clip_interaction",
                )
            )
            return results
        if clip and (clip_value is None or float(clip_value) <= 0):
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="amp_grad_clip_interaction",
                    message=(
                        "AMP + enable_gradient_clipping=true but gradient_clip_value "
                        f"is {clip_value!r}. The AMP scaler will unscale_ once for "
                        "clipping; if no positive clip threshold is set, the wrapper "
                        "may attempt unscale_ a second time before update() — the "
                        "exact pattern that produced 6 smoke failures last run."
                    ),
                    severity="warning",
                    category="amp_grad_clip_interaction",
                    yaml_keys=[
                        "optimization.precision.enabled",
                        "optimization.gradient.clip.enabled",
                        "optimization.gradient.clip.value",
                    ],
                    fix_hint=(
                        "Set optimization.gradient.clip.value to a positive number "
                        "(e.g. 1.0), OR disable optimization.gradient.clip.enabled, "
                        "OR disable optimization.precision.enabled."
                    ),
                )
            )
        else:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="amp_grad_clip_interaction",
                    message="AMP + gradient clipping settings look consistent.",
                    severity="info",
                    category="amp_grad_clip_interaction",
                )
            )
        return results

    # ------------------------------------------------------------------
    # Tier-1 warnings-and-fallbacks additions (audit-ladder extension):
    # - check_early_stopping_metric_compatibility
    # - check_metric_channel_compatibility
    # - check_paradigm_required_fields
    #
    # ``check_legacy_schema_mixing`` used to live here. Deleted (#933): all
    # four ``_LEGACY_CONFLICTS`` legacy leaves resolved to ``None`` on a
    # resolved ``TrainingSettings`` (raw-document question asked of a
    # post-fold object), and the premise was obsolete besides —
    # ``enable_image_normalization`` (whether) and ``normalization_type``
    # (which) are not rivals, so declaring both is ordinary usage. A
    # repointed version would fire at severity="error" on legitimate
    # canonical configs, since post-fold the settings object cannot
    # distinguish "user set both" from "user set one".
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_dotted(config: TrainingSettings, path: tuple[str, ...]) -> Any:
        node: Any = config
        for key in path:
            if node is None:
                return None
            node = getattr(node, key, None)
            if node is None:
                return None
        return node

    def check_early_stopping_metric_compatibility(
        self, config: TrainingSettings
    ) -> list[HealthCheckResult]:
        """Flag ``early_stopping.metric`` that almost-certainly won't appear in val.

        We can't enumerate the strategy's full validation metric set
        without instantiating it, but we can spot the single most common
        bug from the smoke run: a cascading-resolution / multi-scale
        validation pipeline emits ``val_<metric>_<level>x`` keys (e.g.
        ``val_mse_2x``) but the YAML asks for the un-suffixed form.
        """
        results: list[HealthCheckResult] = []
        es_cfg = getattr(config, "early_stopping", None)
        if es_cfg is None or not getattr(es_cfg, "enabled", False):
            return results
        metric = getattr(es_cfg, "metric", None)
        if not metric:
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="early_stopping_metric_compatibility",
                    message="early_stopping.enabled=true but early_stopping.metric is empty.",
                    severity="error",
                    category="early_stopping_metric",
                    yaml_keys=["early_stopping.metric", "early_stopping.enabled"],
                    fix_hint="Set early_stopping.metric to a key produced by validation.",
                )
            )
            return results

        # Heuristic: cascading SR / multi-scale strategies append _Nx.
        cascading = getattr(
            getattr(config, "training", None), "cascading_super_resolution", None
        ) or getattr(getattr(config, "data", None), "multi_scale", None)
        if cascading and not any(metric.endswith(f"_{n}x") for n in (2, 3, 4, 5, 6, 8)):
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="early_stopping_metric_compatibility",
                    message=(
                        f"early_stopping.metric={metric!r} but the strategy emits "
                        f"cascading-scale keys like {metric}_2x. Early stopping "
                        f"will never trigger because the metric never appears."
                    ),
                    severity="warning",
                    category="early_stopping_metric",
                    yaml_keys=["early_stopping.metric"],
                    fix_hint=(
                        f"Set early_stopping.metric to the actual key (e.g. "
                        f"'{metric}_2x') OR disable cascading_super_resolution."
                    ),
                )
            )
        if not results:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="early_stopping_metric_compatibility",
                    message=f"early_stopping.metric={metric!r} is compatible with "
                    "the validation pipeline.",
                    severity="info",
                    category="early_stopping_metric",
                )
            )
        return results

    @staticmethod
    def _expected_outcome_results(meta: object) -> list[HealthCheckResult]:
        """``metadata.expected_outcome`` reads a result against ``metadata.baseline``.

        ``comparable`` stands on its own; ``floor`` and ``ceiling`` describe the
        arm's expected position relative to a comparator, so they need one. A
        floor with no baseline is a marker nothing can check.
        """
        outcome = getattr(meta, "expected_outcome", None)
        if outcome is None:
            return []
        baseline = getattr(meta, "baseline", None)
        if outcome in ("floor", "ceiling") and not baseline:
            return [
                HealthCheckResult(
                    passed=False,
                    check_name="scientific_metadata_expected_outcome",
                    message=(
                        f"metadata.expected_outcome={outcome!r} names a position relative to a "
                        "comparator, and metadata.baseline is not declared."
                    ),
                    severity="warning",
                    fix_hint=(
                        "Declare metadata.baseline (the metadata.name of the arm this one is "
                        "a floor/ceiling of), or set expected_outcome to 'comparable'."
                    ),
                )
            ]
        against = f" against baseline {baseline!r}" if baseline else ""
        return [
            HealthCheckResult(
                passed=True,
                check_name="scientific_metadata_expected_outcome",
                message=f"metadata.expected_outcome={outcome!r}{against}.",
                severity="info",
            )
        ]

    def check_scientific_metadata(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Guard the headline scientific-coherence metadata (2026-06 validation campaign).

        Active only when ``metadata.primary_metric`` is set (the first-class field
        added alongside ``hypothesis`` / ``baseline`` in ``ExperimentMetadataSchema``),
        so configs that don't use it are unaffected — forward-looking prevention of
        the *metric-mismatch* failure mode (179 findings in the inprogress sweep): a
        declared headline metric that the run never actually computes, or that
        disagrees with the metric used for model selection.
        """
        import re as _re

        results: list[HealthCheckResult] = []
        meta = getattr(config, "metadata", None)
        # TrainingSettings.metadata is ExperimentMetadataSchema (mounted 2026-09-02);
        # the dict shape it used to carry is gone with it.
        results.extend(self._expected_outcome_results(meta))
        primary = getattr(meta, "primary_metric", None)
        if not primary:
            return results

        def _base(name: object) -> str:
            s = str(name)
            if s.startswith("val_"):
                s = s[4:]
            return _re.sub(r"_\d+x$", "", s)

        val_cfg = getattr(config, "validation", None)
        declared = list((val_cfg.scoring.compute if val_cfg else None) or [])
        declared_bases = {_base(m) for m in declared}

        # ADVISORY only (severity="info", passed=True so it never gates the audit):
        # metadata.primary_metric is a free-form human label, often a loose family
        # name ("psnr") vs the precise emitted key ("val_robust_mri_psnr"), so a
        # hard equality check produces benign false positives (~34/148 configs in
        # the inprogress sweep). We surface a coherence hint without breaking smoke
        # (avoids the "migration warning breaks configs" trap). Skip prose values.
        primary_base = _base(primary)
        is_token = (" " not in str(primary)) and ("(" not in str(primary))
        not_emitted = bool(is_token and declared_bases and primary_base not in declared_bases)
        if not_emitted:
            message = (
                f"metadata.primary_metric={primary!r} is not in validation.metrics "
                f"({declared}); confirm the strategy actually emits it — otherwise "
                "the declared headline metric has no source (metric-mismatch)."
            )
            fix_hint = (
                "Add the metric to validation.metrics, or set metadata.primary_metric "
                "to a metric the resolved strategy emits."
            )
        else:
            message = f"metadata.primary_metric={primary!r} declared."
            fix_hint = None
        results.append(
            HealthCheckResult(
                passed=True,
                check_name="scientific_metadata_primary_metric",
                message=message,
                severity="info",
                category="scientific_metadata",
                yaml_keys=["metadata.primary_metric", "validation.metrics"],
                fix_hint=fix_hint,
            )
        )
        return results

    def check_vae_pretrain_autoencodes_single_field(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """A ``vae_pretrain`` arm on paired data must autoencode ONE field.

        Guards the 2026-07 ldm_two_stage_ulf_to_hf failure: a stage-1 VAE that
        sets a *translation* direction (``ulf_to_hf`` / ``hf_to_ulf``) on paired
        data trains ``Dec(Enc(input)) → different-field target`` — a degradation
        network, not an autoencoder — which corrupts the frozen stage-2 latent.
        The ``metadata.tags.type == "vae_pretrain"`` tag is the discriminator
        that SPARES the legitimate paired VAE *translators* (``complex_vae`` /
        ``disentangled_vae``), which use ``ulf_to_hf`` on purpose.
        """
        check_name = "vae_pretrain_autoencodes_single_field"
        training = getattr(config, "training", None)
        data = getattr(config, "data", None)
        mode = (getattr(training, "training_mode", "") or "").lower()
        ds_type = (getattr(data, "dataset_type", "") or "").lower()
        paired = {"nifti_paired", "paired_nifti", "paired_mri", "contrast_aware_paired"}
        if mode not in ("vae", "vqvae") or ds_type not in paired:
            return HealthCheckResult(True, check_name, "Not a paired VAE arm — check N/A.", "info")
        meta = getattr(config, "metadata", None)
        tags_val = meta.get("tags") if isinstance(meta, dict) else getattr(meta, "tags", None)
        tag_type = (
            tags_val.get("type") if isinstance(tags_val, dict) else getattr(tags_val, "type", None)
        )
        if str(tag_type or "").lower() != "vae_pretrain":
            return HealthCheckResult(
                True,
                check_name,
                f"Paired VAE tagged {tag_type!r} (not vae_pretrain) — "
                "translation direction is intentional; check N/A.",
                "info",
            )
        bmode = (data.pairing.bidirectional_mode or "").lower()
        if bmode not in ("hf_to_hf", "ulf_to_ulf"):
            return HealthCheckResult(
                False,
                check_name,
                f"vae_pretrain on paired data with bidirectional_mode={bmode!r} "
                "trains Dec(Enc(input)) against a different-field target = a "
                "degradation/translation network, not an autoencoder; this "
                "corrupts the frozen stage-2 autoencoder.",
                "error",
                category="scientific_coherence",
                yaml_keys=[
                    "training.training_mode",
                    "data.bidirectional_mode",
                    "metadata.tags.type",
                ],
                fix_hint=(
                    "Set data.bidirectional_mode: hf_to_hf (autoencode HF) or "
                    "ulf_to_ulf — input≡target, opposite arm dropped."
                ),
            )
        return HealthCheckResult(
            True,
            check_name,
            f"vae_pretrain autoencodes a single field (bidirectional_mode={bmode}).",
            "info",
        )

    # Metrics with channel-count constraints AT THE INCEPTION/VGG INPUT.
    # The actual spectramr.core.metrics.evaluation_metrics implementations
    # auto-adapt channels before forwarding:
    #   - FID  (line 914-917, 941-944) : 1→repeat to 3 ; 3→pass-through.
    #          Anything else (2, 4, 8, …) falls through to InceptionV3
    #          unchanged and DOES error. So enforce {1, 3}.
    #   - LPIPS (line 1465-1476)       : 1→repeat ; 3→pass ; >3→take[:3];
    #          else (e.g. 2)→mean+repeat. Handles every count gracefully.
    # We previously listed FID as {3} only and LPIPS as {1, 3} only —
    # both were stricter than the runtime, producing ~12 false positives
    # per smoke run.
    _METRIC_CHANNEL_CONSTRAINTS: dict[str, tuple[int, ...]] = {
        "compute_fid": (1, 3),
        # compute_lpips intentionally omitted: LPIPS handles any count.
    }

    def check_metric_channel_compatibility(
        self, config: TrainingSettings
    ) -> list[HealthCheckResult]:
        """Flag metrics that won't accept the model's effective output channel count.

        FID expects 3 (RGB), LPIPS expects 1 or 3. The 226 FID failures
        on ``dummy_graph_cold_diffusion`` come from feeding 8-channel
        coil data into FID and getting silent NaN.
        """
        results: list[HealthCheckResult] = []
        metrics_cfg = getattr(config, "metrics", None)
        model_cfg = getattr(config, "model", None)
        if metrics_cfg is None or model_cfg is None:
            return results
        out_channels = int(getattr(model_cfg, "out_channels", 1) or 1)

        for flag, allowed in self._METRIC_CHANNEL_CONSTRAINTS.items():
            if not bool(getattr(metrics_cfg, flag, False)):
                continue
            if out_channels in allowed:
                continue
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="metric_channel_compatibility",
                    message=(
                        f"metrics.{flag}=true but model.out_channels={out_channels}; "
                        f"this metric only accepts {sorted(allowed)} channels and "
                        f"will silently fail / return NaN at validation."
                    ),
                    severity="error",
                    category="metric_channel_compatibility",
                    yaml_keys=[f"metrics.{flag}", "model.out_channels"],
                    fix_hint=(
                        f"Either set metrics.{flag}=false OR add a "
                        f"channel-collapse step (e.g. magnitude / coil RSS) "
                        f"to make the validation tensor {sorted(allowed)} channels."
                    ),
                )
            )
        if not results:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="metric_channel_compatibility",
                    message="All declared metrics are channel-compatible with the model output.",
                    severity="info",
                    category="metric_channel_compatibility",
                )
            )
        return results

    # Per-paradigm required fields. Each entry: (training_mode_substring,
    # required_dotted_paths, suggestion).
    # Paradigm-required fields. Only list fields that have NO sensible
    # schema default — anything with `default=...` in Pydantic is by
    # design and should not fire here. Both the previously-listed
    # `vae.kl_beta_end` (default=1.0) and `diffusion.num_timesteps`
    # (which doesn't even exist — the real field is `diffusion.timesteps`,
    # default=1000) were removed because they fired on every paradigm arm.
    # Per-paradigm required fields list. New entries here MUST point at a
    # field that actually lives in `src/config/schemas/` AND has no sensible
    # default — i.e. the field's absence really does break training.
    #
    # Each entry: (substring matched against strategy_class+training_mode,
    # required dotted paths, fix_hint).
    _PARADIGM_REQUIREMENTS: list[tuple[str, list[tuple[str, ...]], str]] = [
        # DiffusionTrainingStrategy reads `config.training.diffusion.timesteps`
        # at __init__. When the YAML omits the `training.diffusion` block,
        # the field defaults to None and the strategy dies with a misleading
        # `Missing required config: training.diffusion.num_timesteps` error
        # (the path it prints isn't even right — see diffusion.py:103).
        # Surfaced by stage2_ldm_ulf_to_hf in cluster smoke 2026-05-05.
        (
            "diffusion",
            [("training", "diffusion")],
            "Add a 'training.diffusion:' block (timesteps, noise_schedule, "
            "prediction_type). See src/config/schemas/training/base.py "
            "DiffusionTrainingConfigSchema for the full field set.",
        ),
        # VAEStrategy reads `config.training.vae.kl_beta_end` during the
        # KL-warmup schedule. When the `training.vae` block is omitted the
        # strategy silently uses a hard-coded fallback (kl_beta_end=1.0)
        # which produces posterior collapse on configs that expected a
        # higher KL weight. CLAUDE.md pitfall #9 — surface the missing
        # block at audit time instead of letting it fall back silently.
        (
            "vae",
            [("training", "vae")],
            "Add a 'training.vae:' block with kl_beta_end (and other VAE "
            "hyperparameters). See src/config/schemas/training/vae.py "
            "VAETrainingConfigSchema for the full field set.",
        ),
        # F15 (2026-05-21 smoke audit): PINNStrategy reads
        # ``config.losses.pinn`` at __init__ and raises if it's None.
        # The 2026-05-20 smoke caught ``experiment_pinn_csm_estimation``
        # with this — Tier-1 audit must catch it BEFORE the runtime
        # ``[PINN Strategy] config.losses.pinn is None`` crash so the
        # smoke wrapper's audit gate stops the experiment at submission.
        # See TODO/audit/smoke_audit_20260521.md §F11+§F15.
        (
            "pinn",
            [("losses", "pinn")],
            "Add a 'losses.pinn:' block with lambda_pde, "
            "lambda_unit_norm_coil, lambda_magnitude_tv, lambda_pinn_dc "
            "(and their enable flags). See src/config/schemas/loss.py "
            "PINNLossesConfig for the full field set.",
        ),
    ]

    def check_declared_losses_registered(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Reject losses declared in YAML that aren't in LossRegistry.

        An unregistered loss name silently never fires. This is the
        prototype bug behind the 11 ``[CSV] WARNING: Loss 'complex_l1'
        is in CSV fieldnames but NOT in losses_scalar dict`` warnings
        from the last smoke run.
        """
        results: list[HealthCheckResult] = []
        losses_cfg = getattr(config, "losses", None)
        if losses_cfg is None:
            return results

        # Walk every declared loss across every declared list -- from
        # LOSS_LIST_DOMAINS, not a hand-written tuple. This loop (and this
        # comment, which said "the three lists") omitted ``latent_losses``, so a
        # latent arm's losses were never checked for registration at all (#1924).
        all_names: list[tuple[str, str]] = []
        for attr in LOSS_LIST_DOMAINS:
            for entry in getattr(losses_cfg, attr, None) or []:
                # Skip explicitly-disabled losses.
                if getattr(entry, "enabled", True) is False:
                    continue
                name = (
                    getattr(entry, "name", None)
                    or getattr(entry, "loss_type", None)
                    or getattr(entry, "type", None)
                )
                if name:
                    all_names.append((str(name), attr))

        if not all_names:
            return results

        try:
            self._lazy_load_registries()
            registered: set[str] = set(self._loss_registry or set())
        except Exception:
            return results  # registry access failed — caller will see another check

        for name, attr in all_names:
            # ``_loss_registry`` is lowercased and ``LossRegistry.get`` resolves
            # case-insensitively, so compare lowercased — otherwise CamelCase
            # registered aliases (``MMDLoss``, ``SSDULoss``, …) false-fail.
            if name.lower() in registered:
                continue
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="declared_losses_registered",
                    message=(
                        f"losses.{attr} declares {name!r} but no loss is "
                        "registered under that name. The loader will silently "
                        "drop it; the loss will never fire."
                    ),
                    severity="error",
                    category="declared_losses_registered",
                    yaml_keys=[f"losses.{attr}"],
                    fix_hint=(
                        f"Either register {name!r} via @register_loss, OR "
                        "remove the entry from the YAML, OR fix the typo. "
                        f"Available losses: {sorted(registered)[:8]}..."
                        if registered
                        else f"Either register {name!r} via @register_loss OR "
                        "remove the entry from the YAML."
                    ),
                )
            )
        if not results:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="declared_losses_registered",
                    message=f"All {len(all_names)} declared losses are registered.",
                    severity="info",
                    category="declared_losses_registered",
                )
            )
        return results

    def check_paradigm_required_fields(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Reject configs missing paradigm-required fields that today fall back silently."""
        results: list[HealthCheckResult] = []
        training = getattr(config, "training", None)
        if training is None:
            return results
        strategy_class = getattr(training, "strategy_class", "") or ""
        training_mode = getattr(training, "training_mode", "") or ""
        marker = (str(strategy_class) + " " + str(training_mode)).lower()

        for substring, required_paths, hint in self._PARADIGM_REQUIREMENTS:
            if substring not in marker:
                continue
            # The bare 'diffusion' rule must not fire on the SELF-CONTAINED diffusion
            # families, which own their own ``training.<name>`` block and never read
            # ``training.diffusion``. Match EXACT strategy names (not the short
            # substrings 'cold_diffusion'/'guided_diffusion') so a future
            # ``*_guided_diffusion`` that DOES read training.diffusion is not wrongly
            # exempted, and the canonical k-space/graph diffusion arms (which set
            # training.diffusion) still go through the requirement below.
            self_contained_diffusion = (
                "field_cold_diffusion",
                "field_guided_diffusion",
            )
            if (
                substring == "diffusion"
                and any(s in marker for s in self_contained_diffusion)
                and self._resolve_dotted(config, ("training", "diffusion")) is None
            ):
                continue
            for path in required_paths:
                value = self._resolve_dotted(config, path)
                if value is None:
                    results.append(
                        HealthCheckResult(
                            passed=False,
                            check_name="paradigm_required_fields",
                            message=(
                                f"Strategy looks like {substring!r} but "
                                f"required field '{'.'.join(path)}' is unset. "
                                "The loader will fall back to a default that "
                                "may silently break training."
                            ),
                            severity="error",
                            category="paradigm_required_fields",
                            yaml_keys=[".".join(path)],
                            fix_hint=hint,
                        )
                    )
        if not results:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="paradigm_required_fields",
                    message="All paradigm-required fields are present.",
                    severity="info",
                    category="paradigm_required_fields",
                )
            )
        return results

    # ------------------------------------------------------------------
    # Tier-1 audit checks added 2026-05-04 (per smoke-test postmortem spec)
    # ------------------------------------------------------------------

    def check_denoising_model_channels(self, config: TrainingSettings) -> HealthCheckResult:
        """Reject ``model_kwargs.denoising_model.input_channels`` ≠ ``model.in_channels``.

        Surfaced by the 2026-05-03 smoke test as pattern R3 (Laplace/Rician
        diffusion experiments crashed because the denoising sub-network was
        configured for 2-channel input while the data delivered 1 channel).
        """
        model = getattr(config, "model", None)
        if model is None:
            return HealthCheckResult(
                passed=True,
                check_name="denoising_model_channels",
                message="No model section to validate.",
                severity="info",
            )
        kwargs = getattr(model, "model_kwargs", None) or {}
        denoising = kwargs.get("denoising_model") if isinstance(kwargs, dict) else None
        if not isinstance(denoising, dict):
            return HealthCheckResult(
                passed=True,
                check_name="denoising_model_channels",
                message="No denoising_model sub-config; skipping check.",
                severity="info",
            )
        in_ch = getattr(model, "in_channels", None)
        sub_in = denoising.get("input_channels")
        if in_ch is None or sub_in is None:
            return HealthCheckResult(
                passed=True,
                check_name="denoising_model_channels",
                message="Channel fields not declared; skipping.",
                severity="info",
            )
        if int(sub_in) != int(in_ch):
            return HealthCheckResult(
                passed=False,
                check_name="denoising_model_channels",
                message=(
                    f"model.model_kwargs.denoising_model.input_channels={sub_in} "
                    f"does not match model.in_channels={in_ch}. The denoising sub-net "
                    "will receive the wrong tensor shape at runtime."
                ),
                severity="error",
                category="channel_mismatch",
                yaml_keys=[
                    "model.in_channels",
                    "model.model_kwargs.denoising_model.input_channels",
                ],
                fix_hint=(
                    "Set denoising_model.input_channels (and output_channels) "
                    f"to {in_ch} to match model.in_channels."
                ),
            )
        return HealthCheckResult(
            passed=True,
            check_name="denoising_model_channels",
            message="Denoising model channels match parent model.",
            severity="info",
        )

    def check_coil_processing_consistency(self, config: TrainingSettings) -> HealthCheckResult:
        """Reject invalid ``physics.coil_processing`` combinations at audit time.

        Pushes the runtime ``ValueError`` / ``NotImplementedError`` that the
        coil pipeline would otherwise raise to ``spectramr audit`` time (Tier 0/1),
        so a bad config fails before it consumes GPU (CLAUDE.md pitfall #10 —
        fail at audit, not at training). First failure wins:

        1. ``combine.method == "sense"`` needs estimation enabled with a
           non-``none`` method (SENSE combine requires sensitivity maps).
        2. ``compression.method == "gcc"`` is reserved / not implemented.
        3. ``estimation.method == "file"`` needs ``maps_path``.
        4. ``compression.method == "svd"`` ⇒ ``model.in_channels ==
           2 * num_virtual_coils`` (real+imag per virtual coil), skipped for
           models that concatenate extra channels at runtime.
        """
        physics = getattr(config, "physics", None)
        coil = getattr(physics, "coil_processing", None)
        if coil is None:
            return HealthCheckResult(
                passed=True,
                check_name="coil_processing_consistency",
                message="No physics.coil_processing block to validate.",
                severity="info",
            )
        compression = getattr(coil, "compression", None)
        estimation = getattr(coil, "estimation", None)
        combine = getattr(coil, "combine", None)
        comp_method = getattr(compression, "method", "none")
        est_method = getattr(estimation, "method", "none")
        est_enabled = getattr(estimation, "enabled", False)
        comb_method = getattr(combine, "method", "rss")

        # 1. SENSE combine needs sensitivity maps.
        if comb_method == "sense" and not (est_enabled and est_method != "none"):
            return HealthCheckResult(
                passed=False,
                check_name="coil_processing_consistency",
                message=(
                    "physics.coil_processing.combine.method='sense' requires "
                    "estimation enabled with a non-'none' method (SENSE combine "
                    "needs sensitivity maps)."
                ),
                severity="error",
                category="coil_processing",
                yaml_keys=[
                    "physics.coil_processing.combine.method",
                    "physics.coil_processing.estimation.enabled",
                    "physics.coil_processing.estimation.method",
                ],
                fix_hint=(
                    "Set estimation.enabled=true and estimation.method to one of "
                    "power_iter/espirit/pinn/rss/file, or use combine.method='rss'."
                ),
            )

        # 2. GCC compression is reserved.
        if comp_method == "gcc":
            return HealthCheckResult(
                passed=False,
                check_name="coil_processing_consistency",
                message=(
                    "physics.coil_processing.compression.method='gcc' is reserved "
                    "but not implemented; the pipeline raises NotImplementedError."
                ),
                severity="error",
                category="coil_processing",
                yaml_keys=["physics.coil_processing.compression.method"],
                fix_hint="Use compression.method='svd' or 'none'.",
            )

        # 3. file estimation needs a maps_path.
        if est_method == "file" and not getattr(estimation, "maps_path", None):
            return HealthCheckResult(
                passed=False,
                check_name="coil_processing_consistency",
                message=(
                    "physics.coil_processing.estimation.method='file' requires "
                    "estimation.maps_path."
                ),
                severity="error",
                category="coil_processing",
                yaml_keys=["physics.coil_processing.estimation.maps_path"],
                fix_hint="Set estimation.maps_path to the precomputed-maps file.",
            )

        # 4. SVD compression ⇒ in_channels == 2*num_virtual_coils (skip models
        #    that concatenate extra channels — smaps / paired contrast).
        model = getattr(config, "model", None)
        model_type = getattr(model, "model_type", None)
        if (
            comp_method == "svd"
            and model is not None
            and not self._expects_input_concat(model)
            and not self._is_paired_contrast(config, model_type)
        ):
            in_ch = getattr(model, "in_channels", None)
            nvc = getattr(compression, "num_virtual_coils", None)
            if in_ch is not None and nvc is not None and int(in_ch) != 2 * int(nvc):
                return HealthCheckResult(
                    passed=False,
                    check_name="coil_processing_consistency",
                    message=(
                        f"compression.method='svd' with num_virtual_coils={nvc} "
                        f"implies model.in_channels=={2 * int(nvc)} (real+imag per "
                        f"virtual coil), but in_channels={in_ch}."
                    ),
                    severity="error",
                    category="channel_mismatch",
                    yaml_keys=[
                        "physics.coil_processing.compression.num_virtual_coils",
                        "model.in_channels",
                    ],
                    fix_hint=f"Set model.in_channels to {2 * int(nvc)}.",
                )

        # 5/6. output-representation consistency. A 'magnitude' output is a single
        # real combined image, so it needs a combine; a 'complex' output keeps
        # coil phase, so it is incompatible with the (real-magnitude) 'rss'
        # combine. (combine='sense' is keep-coils at data-load, so complex is OK.)
        output = getattr(coil, "output", None)
        out_ch = getattr(output, "channels", "complex")
        if out_ch == "magnitude" and comb_method == "none":
            return HealthCheckResult(
                passed=False,
                check_name="coil_processing_consistency",
                message=(
                    "physics.coil_processing.output.channels='magnitude' requires "
                    "a combine (rss/sense) — a magnitude image is coil-combined."
                ),
                severity="error",
                category="coil_processing",
                yaml_keys=[
                    "physics.coil_processing.output.channels",
                    "physics.coil_processing.combine.method",
                ],
                fix_hint="Set combine.method to 'rss' (or 'sense'), or use a "
                "non-magnitude output.channels.",
            )
        if out_ch == "complex" and comb_method == "rss":
            return HealthCheckResult(
                passed=False,
                check_name="coil_processing_consistency",
                message=(
                    "physics.coil_processing.output.channels='complex' is "
                    "incompatible with combine.method='rss' (RSS produces a real "
                    "magnitude image, not complex coils)."
                ),
                severity="error",
                category="coil_processing",
                yaml_keys=[
                    "physics.coil_processing.output.channels",
                    "physics.coil_processing.combine.method",
                ],
                fix_hint="Use output.channels 'real_interleaved'/'magnitude' with "
                "rss, or combine.method 'none' with complex.",
            )

        return HealthCheckResult(
            passed=True,
            check_name="coil_processing_consistency",
            message="physics.coil_processing is consistent.",
            severity="info",
        )

    def check_complex_unet_even_channels(self, config: TrainingSettings) -> HealthCheckResult:
        """Backbones that operate on real/imag-stacked complex tensors require even ``in_channels``."""
        model = getattr(config, "model", None)
        if model is None:
            return HealthCheckResult(
                passed=True,
                check_name="complex_unet_even_channels",
                message="No model section.",
                severity="info",
            )
        kwargs = getattr(model, "model_kwargs", None) or {}
        backbone = kwargs.get("backbone_type", "") if isinstance(kwargs, dict) else ""
        complex_backbones = {"complex_unet", "diff_varnet", "diff_varnet_kan"}
        if backbone not in complex_backbones:
            return HealthCheckResult(
                passed=True,
                check_name="complex_unet_even_channels",
                message=f"Backbone {backbone!r} does not require even channels.",
                severity="info",
            )
        in_ch = getattr(model, "in_channels", None)
        if in_ch is None or int(in_ch) % 2 == 0:
            return HealthCheckResult(
                passed=True,
                check_name="complex_unet_even_channels",
                message=f"in_channels={in_ch} is even (real/imag-paired).",
                severity="info",
            )
        return HealthCheckResult(
            passed=False,
            check_name="complex_unet_even_channels",
            message=(
                f"Backbone {backbone!r} expects real/imag-stacked input "
                f"(even number of channels), got in_channels={in_ch}."
            ),
            severity="error",
            category="channel_mismatch",
            yaml_keys=["model.in_channels", "model.model_kwargs.backbone_type"],
            fix_hint=(
                "Use an even in_channels (e.g. 2 for 1 complex coil, "
                "8 for 4 SVD-compressed virtual coils)."
            ),
        )

    def check_model_contract_declared(self, config: TrainingSettings) -> HealthCheckResult:
        """Report (no-gate) whether the model declares its dimension contract.

        Layer-1(a) of the dimension-contract plan. Surfaces, per arm, whether the
        resolved model declares ``spatial_dims`` + a domain in its
        ``ModelCapabilities``. Report-only (``info``) until the high-traffic
        back-fill lands and the corpus is validated — the default-deny promotion
        (undeclared + 3D/complex/multi-coil → error) lives in
        ``check_data_model_compatibility``. See ``docs/dimension_contract.rst``.
        """
        model = getattr(config, "model", None)
        model_type = getattr(model, "model_type", None) if model else None
        if not model_type:
            return HealthCheckResult(
                passed=True,
                check_name="model_contract_declared",
                message="No model.model_type to inspect.",
                severity="info",
            )
        from spectramr.models.registry import MODEL_REGISTRY, get_model_capabilities

        if model_type not in MODEL_REGISTRY:
            # check_registered_model_resolves owns the unregistered case.
            return HealthCheckResult(
                passed=True,
                check_name="model_contract_declared",
                message=f"Model {model_type!r} not in registry; skipped.",
                severity="info",
            )
        from spectramr.models.capabilities import CONTRACT_FIELDS

        caps = get_model_capabilities(model_type)
        if caps is None:
            missing = list(CONTRACT_FIELDS)
        else:
            missing = [f for f in CONTRACT_FIELDS if getattr(caps, f, None) is None]
        if not missing:
            return HealthCheckResult(
                passed=True,
                check_name="model_contract_declared",
                message=f"Model {model_type!r} declares its dimension contract.",
                severity="info",
            )
        return HealthCheckResult(
            passed=True,  # report-only (info); promotion to error is corpus-gated
            check_name="model_contract_declared",
            message=(
                f"Model {model_type!r} leaves dimension-contract fields "
                f"undeclared: {', '.join(missing)}. The audit fail-soft skips "
                "dimension checks for this model until they are declared."
            ),
            severity="info",
            category="dimension_contract",
            yaml_keys=["model.model_type"],
            fix_hint=(
                "Declare spatial_dims + input_domain/output_domain on the "
                "@register_model decorator (see docs/dimension_contract.rst)."
            ),
        )

    @staticmethod
    def _data_is_dangerous(config: TrainingSettings) -> str | None:
        """Return a danger label if the data is 3D / complex / multi-coil, else None.

        "Dangerous" = the data shape classes where an *undeclared* model contract
        most often crashes mid-training (the 3D-into-2D conv, the complex/k-space
        channel-layout confusion). Reuses ``spec_card._derive_data_form`` so the
        derivation (incl. the coil-combine→image collapse) matches the rest of the
        ladder exactly.
        """
        from .spec_card import _derive_data_form

        df = _derive_data_form(config)
        if df.get("spatial_dims") == 3:
            return "3D"
        if df.get("domain_inferred") in ("kspace", "complex_image"):
            return "complex/k-space"
        return None

    def check_dangerous_data_requires_contract(self, config: TrainingSettings) -> HealthCheckResult:
        """Default-deny: an undeclared model on dangerous data is flagged.

        Layer-1(b) of the dimension-contract plan. Fires only on the genuinely
        dangerous quadrant (undeclared contract AND 3D/complex/multi-coil data),
        which is where the fail-soft skip turns into a runtime crash. Emitted at
        ``info`` during the rollout (the smoke wrapper is ``--strict``, so a
        warning would gate prematurely); promotion to ``error`` is the final,
        corpus-validated step. See ``docs/dimension_contract.rst``.
        """
        model = getattr(config, "model", None)
        model_type = getattr(model, "model_type", None) if model else None
        if not model_type:
            return HealthCheckResult(
                passed=True,
                check_name="dangerous_data_requires_contract",
                message="No model.model_type to inspect.",
                severity="info",
            )
        from spectramr.models.registry import MODEL_REGISTRY, get_model_capabilities

        if model_type not in MODEL_REGISTRY:
            return HealthCheckResult(
                passed=True,
                check_name="dangerous_data_requires_contract",
                message=f"Model {model_type!r} not in registry; skipped.",
                severity="info",
            )
        from spectramr.models.capabilities import CONTRACT_FIELDS

        # Ask for a DIMENSION CONTRACT specifically, not merely "any declared
        # capability field". This used to read
        # ``get_model_capabilities(model_type) is not None``, which was a proxy
        # -- correct only while ModelCapabilities held nothing but contract
        # fields, and already at odds with this branch's own message. #1916
        # moved the two conditioning flags onto the dataclass, and under the
        # old proxy the 9 models that declare a conditioning flag and no
        # dimension contract would have silently dropped out of this
        # default-deny bucket. Same constant as ``model_contract_declared``
        # (non-negotiable 17).
        declared = get_model_capabilities(model_type)
        if declared is not None and any(
            getattr(declared, f, None) is not None for f in CONTRACT_FIELDS
        ):
            return HealthCheckResult(
                passed=True,
                check_name="dangerous_data_requires_contract",
                message=f"Model {model_type!r} declares a dimension contract.",
                severity="info",
            )
        danger = self._data_is_dangerous(config)
        if danger is None:
            return HealthCheckResult(
                passed=True,
                check_name="dangerous_data_requires_contract",
                message="Data is trivial (2D real image); undeclared model allowed.",
                severity="info",
            )
        return HealthCheckResult(
            passed=True,  # info during rollout; becomes error after corpus validation
            check_name="dangerous_data_requires_contract",
            message=(
                f"Model {model_type!r} declares no dimension contract "
                f"(ModelCapabilities all-None) but data is {danger}. Once the "
                "rollout promotes this check it will be a hard error — an "
                "unaudited model must not run a 3D/complex/multi-coil arm."
            ),
            severity="info",
            category="dimension_contract",
            yaml_keys=[
                "model.model_type",
                "data.sampling.patch_size",
                "data.dataset_type",
                "data.coil_processing_mode",
            ],
            fix_hint=(
                "Back-fill spatial_dims (+ input_domain/output_domain) on this "
                "model's @register_model decorator."
            ),
        )

    def check_metric_domain_matches_loss_output(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """Flag a metric domain that conflicts with the loss output domain — unbridged.

        Layer-1(c) of the dimension-contract plan. A k-space loss output graded by
        image-domain metrics is fine *iff* a transform (e.g. ``ifft_sense_adjoint``)
        bridges the domains first — so this check ONLY flags a declared
        ``metrics.domain`` that differs from ``losses.output_domain`` with no
        transform that will actually run. Conservative + info-level to avoid
        false-positives on the common transform-bridged pattern.

        "Will actually run" is asked of ``resolve_metric_transform`` — the same
        resolver ``MetricsMixin._apply_metric_transforms`` dispatches from. It
        used to be asked of ``bool(metrics.transform)``, which is how 236 arms
        bought an exemption from this check with a knob that had no reader: the
        declaration granted the pass, and then no transform bridged anything
        (#931). A name the dispatcher cannot execute is now its own failure —
        that config raises at validation time rather than grading the wrong
        quantity in silence (pitfall #9).
        """
        losses = getattr(config, "losses", None)
        loss_out = losses.policy.output_domain if losses else None
        metrics = getattr(config, "metrics", None)
        metric_domain = getattr(metrics, "domain", None) if metrics else None
        validation = getattr(config, "validation", None)
        resolution = resolve_metric_transform(validation, metrics)

        # Every declaration, not just the precedence winner. This check is about
        # ``metrics.domain``, and the path that reads ``metrics.domain`` is the
        # same one that dispatches ``metrics.transform`` -- so that key DOES
        # bridge here even though the validation path ignores it. What it may
        # not do is bridge with a name no branch implements.
        declared = declared_metric_transforms(validation, metrics)
        undispatchable = [
            (source, name) for source, name in declared if name not in IMPLEMENTED_METRIC_TRANSFORMS
        ]
        if undispatchable:
            return HealthCheckResult(
                passed=False,
                check_name="metric_domain_matches_loss_output",
                message=(
                    "Metric transform "
                    + ", ".join(f"{n!r} at {s}" for s, n in undispatchable)
                    + " is not implemented — no branch dispatches it, so it "
                    "silently graded the untransformed tensors. Implemented: "
                    f"{sorted(IMPLEMENTED_METRIC_TRANSFORMS)}."
                ),
                severity="error",
                category="dimension_contract",
                yaml_keys=[source for source, _ in undispatchable],
                fix_hint=(
                    "Remove the key if the output is already in the metric "
                    f"domain, or name one of {sorted(IMPLEMENTED_METRIC_TRANSFORMS)}. "
                    "Do NOT assume ifft_mag_combine meant ifft_magnitude: on an "
                    "image-domain output an IFFT produces a Fourier magnitude, "
                    "not a coil combine."
                ),
            )

        bridged = bool(declared) and not resolution.suppressed
        if bridged and loss_out and metric_domain and str(metric_domain) == str(loss_out):
            # A transform firing where the domains ALREADY agree is not a bridge
            # — it moves the metric OUT of the domain both sides declared. The
            # inverse of the facade this check used to grant, and the failure
            # mode that made aliasing ifft_mag_combine unsafe for 112 arms.
            return HealthCheckResult(
                passed=True,  # info during rollout, like the mismatch branch
                check_name="metric_domain_matches_loss_output",
                message=(
                    f"Metric domain and loss output domain already agree "
                    f"({loss_out}), yet a transform still runs "
                    + ", ".join(f"{n!r} at {s}" for s, n in declared)
                    + " — it moves the metric out of the domain both sides "
                    "declared."
                ),
                severity="info",
                category="dimension_contract",
                yaml_keys=[source for source, _ in declared] + ["metrics.domain"],
                fix_hint=("Drop the transform, or correct whichever domain declaration is wrong."),
            )

        if not loss_out or not metric_domain or bridged:
            # No declared conflict, or a transform bridges the two domains.
            return HealthCheckResult(
                passed=True,
                check_name="metric_domain_matches_loss_output",
                message="Metric/loss domains consistent or bridged by a transform.",
                severity="info",
            )
        if str(metric_domain) == str(loss_out):
            return HealthCheckResult(
                passed=True,
                check_name="metric_domain_matches_loss_output",
                message=f"Metric domain matches loss output domain ({loss_out}).",
                severity="info",
            )
        return HealthCheckResult(
            passed=True,  # info during rollout
            check_name="metric_domain_matches_loss_output",
            message=(
                f"Metric domain {metric_domain!r} differs from loss output domain "
                f"{loss_out!r} with no transform to bridge them — metrics may "
                "grade the wrong physical quantity."
            ),
            severity="info",
            category="dimension_contract",
            yaml_keys=[
                "metrics.domain",
                "validation.scoring.output_transform",
                "losses.output_domain",
            ],
            fix_hint=(
                "Set validation.scoring.output_transform (e.g. ifft_sense_adjoint) "
                "to bridge k-space→image, or align metrics.domain with "
                "losses.output_domain."
            ),
        )

    def check_patch_size_power_of_two(self, config: TrainingSettings) -> HealthCheckResult:
        """Reject patch_size not divisible by ``2 ** len(channel_mult)`` for U-Net-like models."""
        data = getattr(config, "data", None)
        model = getattr(config, "model", None)
        if data is None or model is None:
            return HealthCheckResult(
                passed=True,
                check_name="patch_size_power_of_two",
                message="No data or model section.",
                severity="info",
            )
        kwargs = getattr(model, "model_kwargs", None) or {}
        if not isinstance(kwargs, dict):
            return HealthCheckResult(
                passed=True,
                check_name="patch_size_power_of_two",
                message="model_kwargs is not a dict.",
                severity="info",
            )
        channel_mult = kwargs.get("channel_mult") or kwargs.get("features")
        if not isinstance(channel_mult, (list, tuple)):
            return HealthCheckResult(
                passed=True,
                check_name="patch_size_power_of_two",
                message="No channel_mult / features list to derive downsampling depth.",
                severity="info",
            )
        downsample = 2 ** max(len(channel_mult) - 1, 0)
        patch_size = data.sampling.patch_size
        if patch_size is None:
            return HealthCheckResult(
                passed=True,
                check_name="patch_size_power_of_two",
                message="No patch_size declared.",
                severity="info",
            )
        # patch_size is typically [H, W] or [H, W, D].
        spatial_dims = list(patch_size)[:2]
        bad_dims = [d for d in spatial_dims if int(d) % downsample != 0]
        if bad_dims:
            return HealthCheckResult(
                passed=False,
                check_name="patch_size_power_of_two",
                message=(
                    f"patch_size {spatial_dims} not divisible by downsample factor "
                    f"{downsample} (= 2 ** {len(channel_mult) - 1} levels). "
                    "U-Net forward pass will fail at the bottleneck."
                ),
                severity="error",
                category="shape_mismatch",
                yaml_keys=[
                    "data.sampling.patch_size",
                    "model.model_kwargs.channel_mult",
                ],
                fix_hint=(
                    f"Use a patch_size where every spatial dim is a multiple of "
                    f"{downsample} (e.g. {downsample * 4} or {downsample * 8})."
                ),
            )
        return HealthCheckResult(
            passed=True,
            check_name="patch_size_power_of_two",
            message=(f"patch_size {spatial_dims} divisible by downsample factor {downsample}."),
            severity="info",
        )

    def check_hilbert_square_pow_two(self, config: TrainingSettings) -> HealthCheckResult:
        """Hilbert-2D scan order requires square power-of-2 spatial dims."""
        model = getattr(config, "model", None)
        if model is None:
            return HealthCheckResult(
                passed=True,
                check_name="hilbert_square_pow_two",
                message="No model section.",
                severity="info",
            )
        kwargs = getattr(model, "model_kwargs", None) or {}
        if not isinstance(kwargs, dict):
            return HealthCheckResult(
                passed=True,
                check_name="hilbert_square_pow_two",
                message="model_kwargs is not a dict.",
                severity="info",
            )
        scan_mode = kwargs.get("scan_mode")
        if scan_mode != "hilbert_2d":
            return HealthCheckResult(
                passed=True,
                check_name="hilbert_square_pow_two",
                message=f"scan_mode={scan_mode!r} does not require Hilbert constraints.",
                severity="info",
            )
        spatial_shape = kwargs.get("spatial_shape")
        if not isinstance(spatial_shape, (list, tuple)) or len(spatial_shape) < 2:
            return HealthCheckResult(
                passed=False,
                check_name="hilbert_square_pow_two",
                message=(
                    "scan_mode='hilbert_2d' requires spatial_shape to be a 2D "
                    f"(or higher) list, got {spatial_shape!r}."
                ),
                severity="error",
                category="shape_mismatch",
                yaml_keys=["model.model_kwargs.spatial_shape"],
                fix_hint="Set spatial_shape: [N, N] where N is a power of 2.",
            )
        h, w = int(spatial_shape[0]), int(spatial_shape[1])
        is_pow2 = lambda n: (n > 0) and ((n & (n - 1)) == 0)
        if h != w or not is_pow2(h):
            return HealthCheckResult(
                passed=False,
                check_name="hilbert_square_pow_two",
                message=(
                    f"scan_mode='hilbert_2d' requires square power-of-2 spatial "
                    f"dims; got ({h}, {w})."
                ),
                severity="error",
                category="shape_mismatch",
                yaml_keys=[
                    "model.model_kwargs.spatial_shape",
                    "model.model_kwargs.scan_mode",
                ],
                fix_hint="Use spatial_shape: [N, N] with N in {16, 32, 64, 128, 256}.",
            )
        return HealthCheckResult(
            passed=True,
            check_name="hilbert_square_pow_two",
            message=f"spatial_shape ({h}, {w}) is square power-of-2.",
            severity="info",
        )

    def check_pde_synthetic_datasets(self, config: TrainingSettings) -> HealthCheckResult:
        """``dataset_type='pde_synthetic'`` requires either ``index_path`` or a ``datasets`` list."""
        data = getattr(config, "data", None)
        if data is None:
            return HealthCheckResult(
                passed=True,
                check_name="pde_synthetic_datasets",
                message="No data section.",
                severity="info",
            )
        if getattr(data, "dataset_type", "") != "pde_synthetic":
            return HealthCheckResult(
                passed=True,
                check_name="pde_synthetic_datasets",
                message="Not a pde_synthetic dataset; skipping.",
                severity="info",
            )
        has_index = data.source.index_path is not None
        datasets_list = getattr(data, "datasets", None) or []
        # The pde_synthetic dataset code (dataset_instantiator.py
        # _create_pde_synthetic_dataset) only consumes config.pde_problem,
        # config.num_synthetic_samples, and config.patch_size. The
        # datasets[].path field is unused — keeping it set causes the
        # manifest_loader to mistakenly require an H5 path. So pde_problem
        # alone is sufficient.
        has_pde_problem = bool(getattr(data, "pde_problem", None))
        if (
            has_index
            or has_pde_problem
            or (isinstance(datasets_list, list) and len(datasets_list) > 0)
        ):
            return HealthCheckResult(
                passed=True,
                check_name="pde_synthetic_datasets",
                message=("pde_synthetic has pde_problem, index_path, or datasets entry."),
                severity="info",
            )
        return HealthCheckResult(
            passed=False,
            check_name="pde_synthetic_datasets",
            message=(
                "dataset_type='pde_synthetic' but none of data.pde_problem, "
                "data.index_path, or data.datasets is provided."
            ),
            severity="error",
            category="data_loader",
            yaml_keys=["data.pde_problem", "data.index_path", "data.datasets"],
            fix_hint=("Set data.pde_problem (e.g. 'burgers_1d', 'darcy_2d')."),
        )

    @staticmethod
    def _strategy_expects_undersampling(config: TrainingSettings) -> bool:
        """True when the resolved strategy reconstructs from an undersampled measurement.

        The reconstruction, physics-driven and diffusion families do (and any
        strategy that declares ``applies_undersampling``); generative,
        calibration and simulator-driven strategies do not. An unresolvable
        strategy is another check's finding and counts as expecting one, so
        this check keeps its old reach there.
        """
        try:
            from spectramr.infrastructure.training.strategies.diffusion import (
                DiffusionTrainingStrategy,
            )
            from spectramr.infrastructure.training.strategies.graph_cold_diffusion_strategy import (
                GraphColdDiffusionStrategy,
            )
            from spectramr.infrastructure.training.strategies.physics_driven_strategy import (
                PhysicsDrivenTrainingStrategy,
            )
            from spectramr.infrastructure.training.strategies.reconstruction import (
                ReconstructionTrainingStrategy,
            )
            from spectramr.infrastructure.training.strategy_factory import (
                TrainingStrategyFactory,
            )

            strategy_cls = TrainingStrategyFactory().get_strategy_class(config)
        except Exception:
            return True
        if not isinstance(strategy_cls, type):
            return True
        # EXACT classes for the two reconstruction bases: ReconstructionTrainingStrategy
        # is subclassed as a generic base by calibration, acquisition-learning,
        # QSM and pretraining strategies that never reconstruct from an
        # undersampled measurement (20 arms on 2026-09-02). The diffusion
        # families are matched by subclass: every cold-diffusion variant needs
        # the block.
        if strategy_cls in (ReconstructionTrainingStrategy, PhysicsDrivenTrainingStrategy):
            return True
        if issubclass(strategy_cls, (DiffusionTrainingStrategy, GraphColdDiffusionStrategy)):
            return True
        return any(
            base.__dict__.get("applies_undersampling", False) is True
            for base in strategy_cls.__mro__
        )

    def check_reveal_attribution_preconditions(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """``reveal_attribution`` needs a reverse mode, a mask family and a domain.

        The schema validator can only see inside ``validation.sampling``, so it
        catches the sampler requirement and nothing else. The other three live in
        ``model`` and ``undersampling`` — different top-level blocks — and this
        layer is the one that may read all of them at once.

        Without this the arm audits clean, trains, and raises at its FIRST
        validation. That is not a quiet gap either: every failure mode here is
        architecture-determined rather than data-dependent, so every batch
        raises, ``val_count`` reaches 0, and the F36 guard turns it into a
        run-level error hours in (pitfall 9 — reject at load instead).
        """
        name = "reveal_attribution_preconditions"
        sampling = getattr(getattr(config, "validation", None), "sampling", None)
        if not bool(getattr(sampling, "reveal_attribution", False)):
            return HealthCheckResult(
                passed=True,
                check_name=name,
                message="validation.sampling.reveal_attribution is off.",
                severity="info",
            )

        from spectramr.models.diffusion.reveal_attribution import (
            DIRECTION_DEPENDENT_ACCELERATION_TYPES,
            NON_LINE_ACCELERATION_TYPES,
            PARTITIONED_REVERSE_MODES,
        )

        problems: list[str] = []

        # 1. The reverse mode must write each coefficient once and freeze it.
        kwargs = getattr(getattr(config, "model", None), "model_kwargs", None) or {}
        mode = str(kwargs.get("reverse_sampling_mode", "additive"))
        if mode not in PARTITIONED_REVERSE_MODES:
            problems.append(
                f"model.model_kwargs.reverse_sampling_mode={mode!r} rewrites "
                "coefficients across steps, so no step 'wrote' a given bin "
                f"(need one of {sorted(PARTITIONED_REVERSE_MODES)}). Note "
                "'additive' is the constructor default, so an arm that omits the "
                "key gets it."
            )

        # 2. The mask must be a band of lines. `resolve_line_axis` is the
        #    authority and decides from the plane, but it cannot run until
        #    validation; the family name is what is knowable now.
        accel = getattr(config, "undersampling", None)
        accel_type = str(getattr(accel, "acceleration_type", "") or "").lower()
        direction = getattr(accel, "mask_direction", None)
        if accel_type in NON_LINE_ACCELERATION_TYPES:
            problems.append(
                f"undersampling.acceleration_type={accel_type!r} is a point "
                "pattern or non-Cartesian trajectory; it partitions by reveal "
                "step but has no band of lines to attribute error to."
            )
        elif accel_type in DIRECTION_DEPENDENT_ACCELERATION_TYPES and not direction:
            problems.append(
                f"undersampling.acceleration_type={accel_type!r} is line-structured "
                "only when undersampling.mask_direction names the axis; it is unset."
            )

        # 3. The band selector is a k-space mask, so the prediction must be
        #    k-space. Same resolver the visualization path uses.
        try:
            from spectramr.infrastructure.training.utils.domain_inference import (
                needs_ifft_for_visualization,
            )

            predicts_kspace, _ = needs_ifft_for_visualization(config)
        except Exception:  # pragma: no cover - resolver failure is its own check
            predicts_kspace = True
        if not predicts_kspace:
            problems.append(
                "the model predicts in image space; a k-space band mask cannot "
                "select its coefficients."
            )

        if problems:
            joined = "\n  - ".join(problems)
            return HealthCheckResult(
                passed=False,
                check_name=name,
                message=(
                    "validation.sampling.reveal_attribution: true, but this arm "
                    f"cannot produce a reveal partition:\n  - {joined}\n"
                    "Set reveal_attribution: false, or change the declaration(s) "
                    "above. Left as-is the run trains and then fails at its first "
                    "validation."
                ),
                severity="error",
            )
        return HealthCheckResult(
            passed=True,
            check_name=name,
            message=(
                f"reveal_attribution: true with reverse_sampling_mode={mode!r}, "
                f"acceleration_type={accel_type or '<unset>'!r}, k-space prediction."
            ),
            severity="info",
        )

    def check_acceleration_present(self, config: TrainingSettings) -> HealthCheckResult:
        """k-space datasets must declare an ``acceleration:`` block (no silent 1× fallback)."""
        data = getattr(config, "data", None)
        ds_type = str(getattr(data, "dataset_type", "") or "").lower() if data else ""
        # Match the full k-space family, not just the literal ``"kspace"`` —
        # ``m4raw`` / ``fastmri_kspace`` / ``bart_kspace`` / ``ismrmrd_kspace`` /
        # ``oracle_bssfp`` are all k-space and were silently exempted before.
        is_kspace = "kspace" in ds_type or ds_type in {
            "m4raw",
            "fastmri_knee",
            "bart_kspace",
            "ismrmrd_kspace",
            "oracle_bssfp",
        }
        if data is None or not is_kspace:
            return HealthCheckResult(
                passed=True,
                check_name="acceleration_present",
                message="Not a k-space dataset; acceleration not required.",
                severity="info",
            )
        accel = getattr(config, "undersampling", None)
        workflow = getattr(config, "workflow", None)
        task = str(getattr(workflow, "task", "") or "").lower() if workflow else ""
        if accel is None and task.endswith("denoising"):
            # A declared denoiser (rep[0] -> NEX mean, workflow_baselines/b0) has
            # no undersampling BY DESIGN; the check used to fire on it as if the
            # block had been forgotten (cohort review 2026-09-02).
            return HealthCheckResult(
                passed=True,
                check_name="acceleration_present",
                message="k-space dataset with workflow.task=denoising: no undersampling by design.",
                severity="info",
            )
        if accel is None and task and not task.endswith("reconstruction"):
            # A declared non-reconstruction task (synthesis, super-resolution,
            # translation, calibration, ...) on k-space data needs no block; the
            # workflow declaration is the arm's own statement of what it does.
            return HealthCheckResult(
                passed=True,
                check_name="acceleration_present",
                message=f"k-space dataset with workflow.task={task!r}: no undersampling expected.",
                severity="info",
            )
        if accel is None and not self._strategy_expects_undersampling(config):
            # A VAE, GAN, calibration, pretraining or simulator-driven strategy on
            # k-space data is not a reconstruction from an undersampled
            # measurement; an ``undersampling:`` block on such an arm would be
            # the inert declaration `undersampling_block_is_applied` refuses.
            # One rule for both checks (cohort review 2026-09-02).
            return HealthCheckResult(
                passed=True,
                check_name="acceleration_present",
                message=(
                    "k-space dataset on a strategy that does not reconstruct from an "
                    "undersampled measurement: no undersampling block expected."
                ),
                severity="info",
            )
        if accel is None:
            return HealthCheckResult(
                passed=False,
                check_name="acceleration_present",
                message=(
                    "dataset_type='kspace' but no top-level undersampling: section. "
                    "The data loader will silently default to 1× (no undersampling)."
                ),
                severity="error",
                category="silent_fallback",
                # The block this check READS has always been the canonical
                # `config.undersampling` (above), but its remediation named the
                # pre-2026-08-02 `acceleration:` -- so a user following the fix
                # wrote a retired spelling. It loads (posture="fold"), which is
                # exactly why nothing caught it: the advice was wrong in the
                # one direction that never fails.
                #
                # `yaml_keys` is user-facing, not an auto-fix key: `cli/app.py`
                # prints it (`keys: ...`) and serialises it into the audit JSON.
                yaml_keys=["undersampling"],
                fix_hint=(
                    "Add explicit:\nundersampling:\n  base_acceleration: 4\n  center_fraction: 0.08"
                ),
            )
        return HealthCheckResult(
            passed=True,
            check_name="acceleration_present",
            message="acceleration section is present.",
            severity="info",
        )

    def check_acs_within_center_band(self, config: TrainingSettings) -> HealthCheckResult:
        """Coil-map calibration sanity for k-space power_iter/espirit arms.

        Hard invariant: ``acs_size >= kernel_size`` (a smaller ACS can't fit one
        calibration patch → degenerate maps) → error. The ACS-vs-preserved-center
        relationship (``acs_size <= center_fraction*W``) is surfaced as info:
        train/val calibrate from the dense fully-sampled target (maps are
        acceleration-invariant), so it only bites inference-time input-ACS
        calibration (the g-factor overlap).
        """
        data = getattr(config, "data", None)
        ds_type = str(getattr(data, "dataset_type", "") or "").lower() if data else ""
        is_kspace = "kspace" in ds_type or ds_type in {
            "m4raw",
            "fastmri_knee",
            "bart_kspace",
            "ismrmrd_kspace",
            "oracle_bssfp",
        }
        physics = getattr(config, "physics", None)
        coil = getattr(physics, "coil_processing", None)
        est = getattr(coil, "estimation", None)
        method = str(getattr(est, "method", "") or "").lower() if est else ""
        enabled = bool(getattr(est, "enabled", False)) if est else False
        if not is_kspace or est is None or not enabled or method not in ("power_iter", "espirit"):
            return HealthCheckResult(
                passed=True,
                check_name="acs_within_center_band",
                message="Not a k-space power_iter/espirit coil-estimation arm; n/a.",
                severity="info",
            )
        acs_size = int(getattr(est, "acs_size", 0) or 0)
        kernel_size = int(getattr(est, "kernel_size", 0) or 0)
        if acs_size < kernel_size:
            return HealthCheckResult(
                passed=False,
                check_name="acs_within_center_band",
                message=(
                    f"coil estimation acs_size={acs_size} < kernel_size="
                    f"{kernel_size}: the ACS window cannot fit one calibration "
                    "patch → degenerate coil maps."
                ),
                severity="error",
                category="config_error",
                yaml_keys=["physics.coil_processing.estimation.acs_size"],
                fix_hint="Set acs_size >= kernel_size (e.g. acs_size: 24, kernel_size: 12).",
            )
        accel = getattr(config, "undersampling", None)
        cf = getattr(accel, "center_fraction", None)
        patch = data.sampling.patch_size
        w = (
            max(int(patch[0]), int(patch[1]))
            if isinstance(patch, (list, tuple)) and len(patch) >= 2
            else None
        )
        band = int(cf * w) if (cf and w) else None
        if band is not None and acs_size > band:
            return HealthCheckResult(
                passed=True,
                check_name="acs_within_center_band",
                message=(
                    f"acs_size={acs_size} exceeds the mask center band "
                    f"~int(center_fraction*W)={band}; train/val calibrate from the "
                    "dense target (ok), but inference-time input-ACS calibration "
                    "would include aliased lines — raise center_fraction or lower "
                    "acs_size."
                ),
                severity="info",
            )
        return HealthCheckResult(
            passed=True,
            check_name="acs_within_center_band",
            message="Coil calibration ACS region supports the kernel.",
            severity="info",
        )

    def check_nr_metrics_research_mode(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Tier-1: the no-reference (NR) metric battery is research-mode / offline-only.

        Spec §6, scoped to the research-mode contract: NR metrics are
        validation-pending and run ONLY through the offline meta-evaluation
        harness (``spectramr meta-evaluate --nr-battery``) — never during
        training. This check (a) catches a typo'd metric key (error) and
        (b) warns if a *training* config lists NR metrics in
        ``metrics.nr.enabled_metrics``, since they will NOT be computed during
        training (avoids the silent-no-op of pitfall #15 — the knob is read,
        validated, and its research-mode status surfaced).
        """
        results: list[HealthCheckResult] = []
        metrics_cfg = getattr(config, "metrics", None)
        nr_cfg = getattr(metrics_cfg, "nr", None) if metrics_cfg is not None else None
        if nr_cfg is None:
            return results
        enabled = tuple(getattr(nr_cfg, "enabled_metrics", ()) or ())
        if not enabled:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="nr_metrics_research_mode",
                    message="No NR metrics listed; nothing to validate.",
                    severity="info",
                )
            )
            return results

        from spectramr.core.metrics.registry import MetricsRegistry

        unknown = [k for k in enabled if not MetricsRegistry.is_registered(k)]
        if unknown:
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="nr_metrics_research_mode",
                    message=f"metrics.nr.enabled_metrics lists unregistered metrics: {sorted(unknown)}",
                    severity="error",
                    category="silent_fallback",
                    yaml_keys=["metrics.nr.enabled_metrics"],
                    fix_hint="Remove the typo or import the module that registers it.",
                )
            )

        # Research-mode warning: NR metrics never run during training.
        results.append(
            HealthCheckResult(
                passed=True,
                check_name="nr_metrics_research_mode",
                message=(
                    f"{len(enabled)} NR metric(s) listed in metrics.nr.enabled_metrics "
                    "are RESEARCH-MODE / validation-pending and are NOT computed during "
                    "training — they run only via the offline harness "
                    "`spectramr meta-evaluate --nr-battery`."
                ),
                severity="warning",
                category="silent_fallback",
                yaml_keys=["metrics.nr.enabled_metrics", "metrics.nr.research_mode"],
                fix_hint=(
                    "Expected for research configs. To validate the battery run "
                    "`spectramr meta-evaluate --nr-battery`; these metrics do not "
                    "affect training/early-stopping."
                ),
            )
        )
        return results

    def check_val_batch_size(self, config: TrainingSettings) -> HealthCheckResult:
        """Warn if ``validation.val_batch_size`` is missing on a >50M-param model."""
        validation = getattr(config, "validation", None)
        model = getattr(config, "model", None)
        if validation is None or model is None:
            return HealthCheckResult(
                passed=True,
                check_name="val_batch_size",
                message="No validation or model section.",
                severity="info",
            )
        if (validation.loader.batch_size if validation else None) is not None:
            return HealthCheckResult(
                passed=True,
                check_name="val_batch_size",
                message="val_batch_size explicitly set.",
                severity="info",
            )
        # Heuristic: if the model has 'features' or large channel_mult, warn.
        kwargs = getattr(model, "model_kwargs", None) or {}
        features = kwargs.get("features", []) if isinstance(kwargs, dict) else []
        feature_sum = sum(int(f) for f in features) if isinstance(features, (list, tuple)) else 0
        if feature_sum > 1024:
            return HealthCheckResult(
                passed=False,
                check_name="val_batch_size",
                message=(
                    "Large model (sum of feature channels > 1024) without explicit "
                    "validation.val_batch_size — validation may OOM on the same GPU."
                ),
                severity="warning",
                category="oom_risk",
                yaml_keys=["validation.val_batch_size"],
                fix_hint="Add: validation.val_batch_size: 1 or 2.",
            )
        return HealthCheckResult(
            passed=True,
            check_name="val_batch_size",
            message="Model is small; val_batch_size default OK.",
            severity="info",
        )

    def check_multi_contrast_model_support(self, config: TrainingSettings) -> HealthCheckResult:
        """``data.multi_contrast.enabled`` requires contrast-aware *consumers*.

        Pattern C (per-sample FiLM conditioning) is opt-in via
        ``data.multi_contrast.enabled: true``. Every model the batch reaches
        must declare ``supports_contrast_conditioning=True`` on its
        ``@register_model`` decorator -- otherwise ``contrast_idx`` is emitted
        on every batch and silently dropped, exactly the silent fallback this
        audit ladder exists to prevent.

        **Two consumers, not one.** Until 2026-09-07 this check read only
        ``model.model_type``, so an arm passed while doing precisely what the
        check forbids: ``experiment_11_sense_bridge_critic`` pairs a
        contrast-aware generator (``kspace_cold_diffusion``, flag ``True``)
        with a critic (``sense_bridge_patchgan``, flag ``False``) and audited
        clean (#1931). The critic is resolved from
        ``model.discriminator_component.name`` -- the same accessor
        ``ModelBuilder.build_discriminator`` uses, and the only spelling the
        corpus uses (23 arms; the legacy ``model.discriminator`` and
        ``training.gan.discriminator`` layouts are used by 0 arms each and are
        not declared schema fields, so ``extra="ignore"`` drops them anyway).

        **Why the critic's unregistered branch FAILS where the generator's
        defers.** An unknown ``model.model_type`` is deferred to
        ``check_model_registry`` -- but that check reads
        ``config.model.model_type`` only, and no check in this ladder
        validates ``discriminator_component.name``. Deferring would defer to
        nobody, and passing would infer support from absence (non-negotiable
        18: absent is a state to report, never a state to infer). Measured
        2026-09-07: ``MODEL_REGISTRY`` (which owns the capability flags) and
        ``ModelFactory``'s ``ModelRegistry`` (which owns buildability) have
        divergent membership in both directions, so absence here is genuinely
        "unknown", not "unsupported" (#1932). 4 corpus arms name a critic absent
        from ``MODEL_REGISTRY`` yet buildable (``patch_gan_discriminator`` x3,
        ``patchgan`` x1); all 4 have ``multi_contrast`` off, so this branch
        costs 0 arms today.

        Both consumers are reported in **one** result. A chain of early
        returns would surface the generator only, and the user would fix it,
        re-run, and meet the critic on the second pass.
        """
        data = getattr(config, "data", None)
        mc = getattr(data, "multi_contrast", None) if data is not None else None
        if mc is None or not getattr(mc, "enabled", False):
            return HealthCheckResult(
                passed=True,
                check_name="multi_contrast_model_support",
                message="data.multi_contrast disabled; skipping.",
                severity="info",
            )

        model = getattr(config, "model", None)
        model_type = getattr(model, "model_type", None) if model is not None else None
        if not model_type:
            return HealthCheckResult(
                passed=False,
                check_name="multi_contrast_model_support",
                message="data.multi_contrast.enabled=True but model.model_type is missing.",
                severity="error",
                category="silent_fallback",
                yaml_keys=["model.model_type", "data.multi_contrast.enabled"],
                fix_hint=(
                    "Set model.model_type to a contrast-conditioning model "
                    "(e.g. 'geo_mamba_unet' or 'latent_gan_generator')."
                ),
            )

        # Lazy import to avoid circular/registry-bootstrap issues.
        try:
            from spectramr.models.init_registry import (
                populate_model_registry,
            )
            from spectramr.models.registry import (
                MODEL_REGISTRY,
                list_models_with_capability,
                model_supports,
            )

            populate_model_registry()
        except Exception as exc:
            return HealthCheckResult(
                passed=True,
                check_name="multi_contrast_model_support",
                message=f"Model registry unavailable; skipping support check ({exc}).",
                # A ``passed=True`` result is filtered out of ``report.warnings``
                # (which requires ``not passed``), so labelling this "warning" made
                # it an invisible non-warning. It is a genuine SKIP — label it
                # "info" so it is not a phantom warning that never surfaces.
                severity="info",
            )

        problems: list[str] = []
        yaml_keys: list[str] = ["data.multi_contrast.enabled"]
        hints: list[str] = []
        notes: list[str] = []

        # ---- consumer 1: the generator (``model.model_type``) ----
        if model_type not in MODEL_REGISTRY:
            # check_model_registry handles this case explicitly elsewhere.
            notes.append(
                f"model_type={model_type!r} not in registry; deferred to check_model_registry"
            )
        elif model_supports(model_type, "supports_contrast_conditioning"):
            notes.append(f"model_type={model_type!r} supports contrast conditioning")
        else:
            compatible = list_models_with_capability("supports_contrast_conditioning")
            problems.append(
                f"model_type={model_type!r} does not declare supports_contrast_conditioning=True"
            )
            yaml_keys.append("model.model_type")
            hints.append(
                f"For the generator: either set data.multi_contrast.enabled=False, "
                f"or pick one of: "
                f"{compatible if compatible else '<no models declare this capability>'}. "
                f"To add support to a new model, set "
                f"@register_model(..., supports_contrast_conditioning=True) "
                f"and accept a `contrast_idx` kwarg in its forward()."
            )

        # ---- consumer 2: the critic (``model.discriminator_component.name``) ----
        # ``ModelComponentSchema.name`` defaults to ``""``, not ``None``, so an
        # undeclared critic must be tested as falsy rather than ``is None``.
        disc_component = getattr(model, "discriminator_component", None)
        disc_name = getattr(disc_component, "name", None) if disc_component is not None else None
        if not disc_name:
            notes.append("no discriminator declared")
        elif disc_name not in MODEL_REGISTRY:
            # NOT deferred, and NOT passed -- see the docstring. Nothing else
            # validates this key, so silence here would be manufactured.
            problems.append(
                f"discriminator_component.name={disc_name!r} is absent from "
                f"MODEL_REGISTRY, so its contrast-conditioning capability "
                f"cannot be established"
            )
            yaml_keys.append("model.discriminator_component.name")
            hints.append(
                f"For the critic: {disc_name!r} is not in MODEL_REGISTRY (it may still "
                f"build -- ModelFactory keeps a separate registry -- but no capability "
                f"flags are recorded for it, and no audit check validates this key -- #1932). "
                f"Register it with @register_model(..., supports_contrast_conditioning=...) "
                f"so the declaration is auditable, or set "
                f"data.multi_contrast.enabled=False."
            )
        elif model_supports(disc_name, "supports_contrast_conditioning"):
            notes.append(f"discriminator={disc_name!r} supports contrast conditioning")
        else:
            problems.append(
                f"discriminator_component.name={disc_name!r} does not declare "
                f"supports_contrast_conditioning=True"
            )
            yaml_keys.append("model.discriminator_component.name")
            census = _contrast_aware_critics(MODEL_REGISTRY)
            if census[0]:
                available = f"contrast-aware discriminators available: {census[0]}"
            else:
                available = (
                    f"no registered discriminator declares "
                    f"supports_contrast_conditioning=True (0 of {census[1]} "
                    f"discriminator names), so there is no drop-in replacement"
                )
            hints.append(
                f"For the critic: {available} -- see #1931. Either set "
                "data.multi_contrast.enabled=False, drop "
                "model.discriminator_component, or teach the critic to consume "
                "contrast_idx (accept the kwarg in discriminate()/forward(), then set "
                "@register_model(..., supports_contrast_conditioning=True))."
            )

        if not problems:
            return HealthCheckResult(
                passed=True,
                check_name="multi_contrast_model_support",
                message="; ".join(notes) + ".",
                severity="info",
            )

        return HealthCheckResult(
            passed=False,
            check_name="multi_contrast_model_support",
            message=(
                "data.multi_contrast.enabled=True but "
                + "; ".join(problems)
                + ". contrast_idx will be silently dropped at the model boundary."
            ),
            severity="error",
            category="silent_fallback",
            yaml_keys=yaml_keys,
            fix_hint=" ".join(hints),
        )

    def check_vendor_model_support(self, config: TrainingSettings) -> HealthCheckResult:
        """``data.multi_contrast.vendor_map`` requires a vendor-aware model.

        The vendor / protocol soft-prompt path (multi-contrast Item 4) is
        opt-in via a non-empty ``data.multi_contrast.vendor_map`` (paired
        with ``n_vendors``). When set, the data pipeline emits a
        ``vendor_id`` long tensor on every batch and the strategy is
        expected to feed it through ``VendorPromptEmbedding`` (or a model
        that accepts the kwarg directly). The chosen ``model.model_type``
        must declare ``supports_vendor_conditioning=True`` on its
        ``@register_model`` decorator -- otherwise ``vendor_id`` is
        silently dropped, the same anti-pattern this audit ladder exists
        to prevent. Mirrors ``check_multi_contrast_model_support`` for
        symmetry.
        """
        data = getattr(config, "data", None)
        mc = getattr(data, "multi_contrast", None) if data is not None else None
        vendor_map = getattr(mc, "vendor_map", None) if mc is not None else None
        n_vendors = int(getattr(mc, "n_vendors", 0)) if mc is not None else 0
        if not vendor_map and n_vendors == 0:
            return HealthCheckResult(
                passed=True,
                check_name="vendor_model_support",
                message="data.multi_contrast.vendor_map empty; skipping.",
                severity="info",
            )

        # Consistency check: when both are set, they must agree.
        if vendor_map and n_vendors and len(vendor_map) != n_vendors:
            return HealthCheckResult(
                passed=False,
                check_name="vendor_model_support",
                message=(
                    f"data.multi_contrast.vendor_map has {len(vendor_map)} entries "
                    f"but n_vendors={n_vendors}. The two must agree."
                ),
                severity="error",
                category="schema",
                yaml_keys=[
                    "data.multi_contrast.vendor_map",
                    "data.multi_contrast.n_vendors",
                ],
                fix_hint=(
                    f"Set n_vendors={len(vendor_map)} to match vendor_map, or "
                    "remove vendor_map to disable vendor conditioning."
                ),
            )

        model = getattr(config, "model", None)
        model_type = getattr(model, "model_type", None) if model is not None else None
        if not model_type:
            return HealthCheckResult(
                passed=False,
                check_name="vendor_model_support",
                message=("data.multi_contrast.vendor_map is set but model.model_type is missing."),
                severity="error",
                category="silent_fallback",
                yaml_keys=["model.model_type", "data.multi_contrast.vendor_map"],
            )

        try:
            from spectramr.models.init_registry import (
                populate_model_registry,
            )
            from spectramr.models.registry import (
                MODEL_REGISTRY,
                list_models_with_capability,
                model_supports,
            )

            populate_model_registry()
        except Exception as exc:
            return HealthCheckResult(
                passed=True,
                check_name="vendor_model_support",
                message=f"Model registry unavailable; skipping support check ({exc}).",
                # A ``passed=True`` result is filtered out of ``report.warnings``
                # (which requires ``not passed``), so labelling this "warning" made
                # it an invisible non-warning. It is a genuine SKIP — label it
                # "info" so it is not a phantom warning that never surfaces.
                severity="info",
            )

        if model_type not in MODEL_REGISTRY:
            return HealthCheckResult(
                passed=True,
                check_name="vendor_model_support",
                message=(
                    f"model_type={model_type!r} not in registry; deferred to check_model_registry."
                ),
                severity="info",
            )

        if model_supports(model_type, "supports_vendor_conditioning"):
            return HealthCheckResult(
                passed=True,
                check_name="vendor_model_support",
                message=f"model_type={model_type!r} supports vendor conditioning.",
                severity="info",
            )

        compatible = list_models_with_capability("supports_vendor_conditioning")
        return HealthCheckResult(
            passed=False,
            check_name="vendor_model_support",
            message=(
                f"data.multi_contrast.vendor_map is set but model_type="
                f"{model_type!r} does not declare supports_vendor_conditioning=True. "
                "vendor_id will be silently dropped at the model boundary."
            ),
            severity="error",
            category="silent_fallback",
            yaml_keys=[
                "data.multi_contrast.vendor_map",
                "model.model_type",
            ],
            fix_hint=(
                (
                    f"Either remove data.multi_contrast.vendor_map (and "
                    f"n_vendors), or pick one of: {compatible}."
                )
                if compatible
                else (
                    "Vendor conditioning is NOT IMPLEMENTED anywhere in the "
                    "framework yet -- this is a framework gap, not a wrong "
                    "model choice, so no model_type will satisfy this check. "
                    "Zero of the registered models declare "
                    "supports_vendor_conditioning, no model's forward() or "
                    "__init__ takes a vendor argument, and "
                    "VendorPromptEmbedding "
                    "(models/blocks/vendor_prompt_embedding.py) has zero "
                    "instantiations. Remove data.multi_contrast.vendor_map "
                    "(and n_vendors) to proceed. To BUILD the capability: "
                    "wire VendorPromptEmbedding into a generator, accept a "
                    "`vendor_id` kwarg in its forward(), and register it with "
                    "@register_model(..., supports_vendor_conditioning=True)."
                )
            ),
        )

    def check_latent_decode_resolution(self, config: TrainingSettings) -> HealthCheckResult:
        """Latent models must define a decoder when target spatial resolution exceeds latent."""
        model = getattr(config, "model", None)
        data = getattr(config, "data", None)
        if model is None or data is None:
            return HealthCheckResult(
                passed=True,
                check_name="latent_decode_resolution",
                message="No model or data section.",
                severity="info",
            )
        model_type = (getattr(model, "model_type", "") or "").lower()
        if "latent" not in model_type:
            return HealthCheckResult(
                passed=True,
                check_name="latent_decode_resolution",
                message="Not a latent model; skipping.",
                severity="info",
            )
        kwargs = getattr(model, "model_kwargs", None) or {}
        if not isinstance(kwargs, dict):
            return HealthCheckResult(
                passed=True,
                check_name="latent_decode_resolution",
                message="model_kwargs is not a dict.",
                severity="info",
            )
        latent_h = kwargs.get("latent_height")
        latent_w = kwargs.get("latent_width")
        patch_size = data.sampling.patch_size
        if patch_size is None or latent_h is None or latent_w is None:
            return HealthCheckResult(
                passed=True,
                check_name="latent_decode_resolution",
                message="Latent or patch dimensions not declared; skipping.",
                severity="info",
            )
        target_h, target_w = int(patch_size[0]), int(patch_size[1])
        if target_h <= int(latent_h) and target_w <= int(latent_w):
            return HealthCheckResult(
                passed=True,
                check_name="latent_decode_resolution",
                message="Latent dims >= target dims; no decode required.",
                severity="info",
            )
        # Validation strategy already calls model.sample() for latent diffusion
        # (Phase B3 fix), so this is a soft warning rather than an error.
        if "diffusion" in model_type:
            return HealthCheckResult(
                passed=True,
                check_name="latent_decode_resolution",
                message=(
                    "Latent diffusion: validation uses model.sample() which decodes "
                    "back to image space (Phase B3 code fix)."
                ),
                severity="info",
            )
        return HealthCheckResult(
            passed=False,
            check_name="latent_decode_resolution",
            message=(
                f"Latent model with latent dims ({latent_h}x{latent_w}) but target "
                f"patch_size ({target_h}x{target_w}) is larger; decoder must be wired up."
            ),
            severity="warning",
            category="shape_mismatch",
            yaml_keys=[
                "model.model_kwargs.latent_height",
                "model.model_kwargs.latent_width",
                "data.sampling.patch_size",
            ],
            fix_hint=(
                "Either expose decoder dims via model_kwargs.decode_to_image_space=True "
                "or shrink data.sampling.patch_size to the latent grid."
            ),
        )

    def check_model_loss_output_domain(self, config: TrainingSettings) -> HealthCheckResult:
        """Reject ``model.output_domain`` ≠ ``losses.output_domain`` mismatches.

        The kspace_cold_diffusion family flooded the cluster smoke tests
        with this exact pattern: a model declared as outputting one
        domain, paired with a losses block declaring a different
        ``output_domain``. The implicit FFT/iFFT auto-bridge in
        :class:`LossBuilder` covers some directions silently, but
        ``model.output_domain == 'kspace'`` against
        ``losses.output_domain == 'image'`` is a real bug that
        produces meaningless gradients.

        Skipped when the model is unannotated (legacy escape hatch) or
        when ``losses.output_domain`` is unset.
        """
        from spectramr.models.capabilities import ModelCapabilities
        from spectramr.models.registry import MODEL_REGISTRY

        model_cfg = getattr(config, "model", None)
        losses_cfg = getattr(config, "losses", None)
        if model_cfg is None or losses_cfg is None:
            return HealthCheckResult(
                passed=True,
                check_name="model_loss_output_domain",
                message="model or losses section absent; skipping.",
                severity="info",
            )
        loss_out = (losses_cfg.policy.output_domain if losses_cfg else None) or None
        if not loss_out:
            return HealthCheckResult(
                passed=True,
                check_name="model_loss_output_domain",
                message="losses.output_domain unset; skipping.",
                severity="info",
            )
        model_type = getattr(model_cfg, "model_type", None)
        entry = MODEL_REGISTRY.get(model_type) if model_type else None
        caps = entry.get("capabilities") if entry else None
        if not isinstance(caps, ModelCapabilities) or caps.output_domain is None:
            return HealthCheckResult(
                passed=True,
                check_name="model_loss_output_domain",
                message=(f"model_type={model_type!r} unannotated; cross-check skipped."),
                severity="info",
            )

        model_out = caps.output_domain
        # Models can declare a tuple of output domains (e.g.
        # cs_mno_operator handles both pde_grid and image). If any
        # element matches the loss-side declaration, we're good.
        model_out_set = set(model_out) if isinstance(model_out, (tuple, list)) else {model_out}
        if loss_out in model_out_set:
            return HealthCheckResult(
                passed=True,
                check_name="model_loss_output_domain",
                message=(
                    f"model.output_domain={model_out!r} matches losses.output_domain={loss_out!r}."
                ),
                severity="info",
            )

        # Auto-bridged patterns:
        # - image-output model + kspace_losses (LossBuilder auto-FFT)
        # - image-output model + complex_losses (LossBuilder auto-cast)
        # - pde_grid ↔ image are tensor-shape equivalent: both are
        #   real-valued [B, C, *spatial] tensors. L1/L2/SSIM work
        #   identically on either. Treating them as compatible avoids
        #   forcing PDE benchmark YAMLs to invent a `losses.output_domain:
        #   pde_grid` value the LossConfigSchema doesn't currently accept.
        bridged = any(
            (mo == "image" and loss_out == "kspace")
            or (mo == "image" and loss_out == "complex_image")
            or (mo == "pde_grid" and loss_out == "image")
            or (mo == "image" and loss_out == "pde_grid")
            for mo in model_out_set
        )
        if bridged:
            return HealthCheckResult(
                passed=True,
                check_name="model_loss_output_domain",
                message=(
                    f"model.output_domain={model_out!r}, "
                    f"losses.output_domain={loss_out!r}: bridged "
                    "automatically by LossBuilder (legacy implicit "
                    "auto-bridge — Phase 4d migration pending)."
                ),
                severity="info",
            )

        return HealthCheckResult(
            passed=False,
            check_name="model_loss_output_domain",
            message=(
                f"model_type={model_type!r} declares output_domain="
                f"{model_out!r} but losses.output_domain={loss_out!r}. "
                "No auto-bridge for this direction; declare an explicit "
                "adapter under `adapters.pre_loss_pred:` or fix the YAML."
            ),
            severity="error",
            category="model_loss_output_domain",
            yaml_keys=["model.model_type", "losses.output_domain"],
            fix_hint=(
                "Either change losses.output_domain to "
                f"{model_out!r} (match the model), or add an "
                "`adapters.pre_loss_pred:` chain that bridges "
                f"{model_out!r} → {loss_out!r}."
            ),
        )

    def check_target_domain_matches_registered_output_domain(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """Reject ``model.target_domain`` ≠ the model's registered output_domain.

        ``model.target_domain`` is Priority-1 in
        :func:`spectramr.infrastructure.training.utils.domain_inference.infer_output_domain`
        — it OVERRIDES the registered ``output_domain`` capability and drives
        :func:`needs_ifft_for_visualization`. A YAML that sets
        ``target_domain: kspace`` on a model registered ``output_domain="image"``
        makes the validation writer IFFT an already-image prediction → k-space
        rendered as an image (DC-blob + concentric rings): the E-VIZ2 smoke
        finding (2026-06-16, exp_p7). The sibling
        :meth:`check_model_loss_output_domain` reads the *registered* domain, not
        ``target_domain``, so it never caught this override (the audit gap).

        Skipped when ``target_domain`` is unset, or when the model is unannotated
        (legacy escape hatch — unannotated models rely on the
        ``KNOWN_*_OUTPUT_MODELS`` sets and cannot be cross-checked here; the
        holistic VF-family domain re-classification is the tracked follow-up).
        """
        from spectramr.models.capabilities import ModelCapabilities
        from spectramr.models.registry import MODEL_REGISTRY

        model_cfg = getattr(config, "model", None)
        if model_cfg is None:
            return HealthCheckResult(
                passed=True,
                check_name="target_domain_output_domain",
                message="model section absent; skipping.",
                severity="info",
            )
        target_domain = getattr(model_cfg, "target_domain", None) or None
        if not target_domain:
            return HealthCheckResult(
                passed=True,
                check_name="target_domain_output_domain",
                message="model.target_domain unset; skipping.",
                severity="info",
            )
        target_domain = str(target_domain).lower()
        model_type = getattr(model_cfg, "model_type", None)
        entry = MODEL_REGISTRY.get(model_type) if model_type else None
        caps = entry.get("capabilities") if entry else None
        if not isinstance(caps, ModelCapabilities) or caps.output_domain is None:
            return HealthCheckResult(
                passed=True,
                check_name="target_domain_output_domain",
                message=(
                    f"model_type={model_type!r} unannotated; target_domain cross-check skipped."
                ),
                severity="info",
            )
        model_out = caps.output_domain
        model_out_set = {
            str(d).lower()
            for d in (model_out if isinstance(model_out, (tuple, list)) else (model_out,))
        }
        if target_domain in model_out_set:
            return HealthCheckResult(
                passed=True,
                check_name="target_domain_output_domain",
                message=(
                    f"model.target_domain={target_domain!r} matches registered "
                    f"output_domain={model_out!r}."
                ),
                severity="info",
            )
        return HealthCheckResult(
            passed=False,
            check_name="target_domain_output_domain",
            message=(
                f"model.target_domain={target_domain!r} overrides the registered "
                f"output_domain={model_out!r} for model_type={model_type!r}. "
                "target_domain is Priority-1 in infer_output_domain and drives "
                "needs_ifft_for_visualization, so this forces a spurious (I)FFT "
                "on the prediction → k-space-rendered-as-image (E-VIZ2)."
            ),
            severity="error",
            category="target_domain_output_domain",
            yaml_keys=["model.target_domain", "model.model_type"],
            fix_hint=(
                f"Set model.target_domain to {model_out!r} (match the registered "
                "output domain) or drop the key (the registry capability is the "
                "SSOT). Only override target_domain when the arm genuinely emits a "
                "different domain than the model registration declares."
            ),
        )

    def check_attention_domain_compatibility(
        self, config: TrainingSettings
    ) -> list[HealthCheckResult]:
        """Reject attention/backbone/domain combinations the model cannot honor.

        The k-space cold-diffusion stack feeds its backbone k-space features
        when ``model_kwargs.force_pure_kspace`` is true and image features
        otherwise. Several attention blocks carry an internal domain
        assumption; the dispatch derives ``feature_domain`` from
        ``force_pure_kspace`` and threads it through so the blocks orient
        their FFTs correctly. This check enforces, at ~100 ms audit time, the
        contracts that would otherwise surface as a build-time crash or (worse)
        a silently-mislabeled arm:

        * **R0** — an ``attention_type`` outside the advertised set.
        * **R1** — ``force_pure_kspace`` + ``backbone_type=unet`` builds
          ``PureKSpaceUNet``, which has NO attention seam, so any attention
          other than ``none`` is silently dropped (pitfall #16 facade).
        * **R2** — ``backbone_type=complex_unet`` requesting an attention the
          block dispatch can't build (``spatial`` exists only in the up-block,
          so it crashes the down block at build time).
        * **R3** — ``backbone_type=complex_unet`` requesting an attention that
          does not support the derived ``feature_domain`` (vacuously green
          today — the ratchet for future single-domain attention blocks).

        Guarded to the ``kspace_cold_diffusion`` model family (and its
        ``kspace_cold_diffusion_generator`` alias); everything else skips with
        an informational pass.
        """
        from spectramr.models.blocks.attention_domains import (
            ATTENTION_DOMAIN_SUPPORT,
            COMPLEX_UNET_BLOCK_ATTENTION,
        )

        name = "attention_domain_compatibility"
        model_cfg = getattr(config, "model", None)
        if model_cfg is None:
            return [
                HealthCheckResult(
                    passed=True,
                    check_name=name,
                    message="model section absent; skipping.",
                    severity="info",
                )
            ]
        model_type = str(getattr(model_cfg, "model_type", "") or "").lower()
        if model_type not in {
            "kspace_cold_diffusion",
            "kspace_cold_diffusion_generator",
        }:
            return [
                HealthCheckResult(
                    passed=True,
                    check_name=name,
                    message=(
                        f"model_type={model_type!r} is not a kspace_cold_diffusion "
                        "arm; attention-domain check skipped."
                    ),
                    severity="info",
                )
            ]

        kw = dict(getattr(model_cfg, "model_kwargs", {}) or {})
        # Mirror the constructor defaults (kspace_cold_diffusion_generator.py):
        # attention_type defaults to "self", backbone_type to "unet".
        attention_type = str(kw.get("attention_type", "self")).lower()
        backbone = str(kw.get("backbone_type", "unet")).lower()
        fpk = bool(kw.get("force_pure_kspace", False))
        feature_domain = "kspace" if fpk else "image"
        keys = [
            "model.model_kwargs.attention_type",
            "model.model_kwargs.backbone_type",
            "model.model_kwargs.force_pure_kspace",
        ]

        # R0 — unknown attention_type.
        supported = ATTENTION_DOMAIN_SUPPORT.get(attention_type)
        if supported is None:
            return [
                HealthCheckResult(
                    passed=False,
                    check_name=name,
                    message=(
                        f"attention_type={attention_type!r} is not an advertised "
                        f"option (valid: {sorted(ATTENTION_DOMAIN_SUPPORT)})."
                    ),
                    severity="error",
                    category="attention_domain",
                    yaml_keys=keys,
                    fix_hint=(
                        "Set model_kwargs.attention_type to one of "
                        f"{sorted(ATTENTION_DOMAIN_SUPPORT)}."
                    ),
                )
            ]

        # R1 — PureKSpaceUNet has no attention seam.
        if fpk and backbone == "unet" and attention_type != "none":
            return [
                HealthCheckResult(
                    passed=False,
                    check_name=name,
                    message=(
                        "force_pure_kspace + backbone_type='unet' builds "
                        "PureKSpaceUNet, which has no attention seam — "
                        f"attention_type={attention_type!r} would be silently "
                        "dropped (pitfall #16 facade)."
                    ),
                    severity="error",
                    category="attention_domain",
                    yaml_keys=keys,
                    fix_hint=(
                        "Set model_kwargs.attention_type: 'none' (PureKSpaceUNet "
                        "runs vanilla), or backbone_type: 'complex_unet' to keep "
                        "block-level attention."
                    ),
                )
            ]

        # R2 / R3 apply to the complex_unet block dispatch.
        if backbone == "complex_unet":
            if attention_type not in COMPLEX_UNET_BLOCK_ATTENTION:
                return [
                    HealthCheckResult(
                        passed=False,
                        check_name=name,
                        message=(
                            f"backbone_type='complex_unet' cannot build "
                            f"attention_type={attention_type!r} — it is not in the "
                            "down/up block dispatch (e.g. 'spatial' exists only in "
                            "the up-block, so ComplexUNet crashes at build time)."
                        ),
                        severity="error",
                        category="attention_domain",
                        yaml_keys=keys,
                        fix_hint=(
                            "Choose an attention_type the complex_unet blocks build: "
                            f"{sorted(COMPLEX_UNET_BLOCK_ATTENTION)}."
                        ),
                    )
                ]
            if feature_domain not in supported:
                return [
                    HealthCheckResult(
                        passed=False,
                        check_name=name,
                        message=(
                            f"attention_type={attention_type!r} does not support "
                            f"feature_domain={feature_domain!r} (derived from "
                            f"force_pure_kspace={fpk}); it supports "
                            f"{sorted(supported)}."
                        ),
                        severity="error",
                        category="attention_domain",
                        yaml_keys=keys,
                        fix_hint=(
                            "Flip force_pure_kspace to feed the supported domain, or "
                            "choose an attention_type that supports "
                            f"{feature_domain!r}."
                        ),
                    )
                ]

        return [
            HealthCheckResult(
                passed=True,
                check_name=name,
                message=(
                    f"attention_type={attention_type!r} on backbone={backbone!r} "
                    f"is compatible with feature_domain={feature_domain!r}."
                ),
                severity="info",
            )
        ]

    def check_data_model_compatibility(self, config: TrainingSettings) -> HealthCheckResult:
        """Cross-check data block ⇆ model registry capabilities.

        Phase 2 of the experiment-spec-card design. Three states:

        - Model unannotated → skip (legacy escape hatch).
        - Annotated and matches data → pass.
        - Annotated and mismatched → declared adapter chain must bridge,
          else error with fix-hint listing candidate adapters.

        Currently checks ``spatial_dims`` and ``input_domain``. Channel
        and complex-tensor checks land alongside the matching adapter
        types in Phase 3.
        """
        # Force-import the adapters package so its concrete adapters
        # register before the chain composer / candidate suggester runs.
        import spectramr.data.adapters  # noqa: F401
        from spectramr.infrastructure.validation.adapter_composition import (
            candidate_adapters_for,
            compose_chain,
            forms_match,
        )
        from spectramr.infrastructure.validation.spec_card import (
            _derive_data_form,
            _derive_model_form,
        )

        data_form = _derive_data_form(config)
        model_form = _derive_model_form(config)

        if not data_form.get("present") or not model_form.get("present"):
            return HealthCheckResult(
                passed=True,
                check_name="data_model_compatibility",
                message="data or model section absent; skipping.",
                severity="info",
            )

        caps = model_form.get("capabilities")
        if caps is None:
            return HealthCheckResult(
                passed=True,
                check_name="data_model_compatibility",
                message=(
                    f"model_type='{model_form.get('model_type')}' is unannotated; "
                    "audit cross-check skipped (legacy escape hatch). Add "
                    "spatial_dims/input_domain to its @register_model decorator "
                    "to enable the strict check."
                ),
                severity="info",
            )

        # 1-D sequence / navigator encoders (caps.spatial_dims == (1,), e.g.
        # ``nav_encoder_1d`` in the IB-VF arm) consume a strategy-extracted 1-D
        # signal (the DC k-space navigator) from the 2-D acquisition; the
        # generic spatial-dims cross-check does not apply. (smoke_audit_20260526)
        if tuple(caps.spatial_dims or ()) == (1,):
            return HealthCheckResult(
                passed=True,
                check_name="data_model_compatibility",
                message=(
                    f"model_type='{model_form.get('model_type')}' is a 1-D "
                    "sequence/navigator encoder (spatial_dims=(1,)); the training "
                    "strategy extracts a 1-D signal from the 2-D acquisition — "
                    "spatial-dims cross-check skipped."
                ),
                severity="info",
            )

        # Build the upstream "what data produces" form and the model's
        # required form.
        upstream = {
            "spatial_dims": data_form.get("spatial_dims"),
            "domain": data_form.get("domain_inferred"),
        }
        required = {
            "spatial_dims": caps.spatial_dims,
            "domain": caps.input_domain,
        }

        # Equivariant Imaging (and the reconstruction-family self-supervised
        # strategies it extends) adjoint k-space to the image domain INTERNALLY
        # via the inherited ``ReconstructionTrainingStrategy._prepare_generator_inputs``
        # (A^* y), so a kspace dataset legitimately feeds an image-domain model
        # without a declared pre_model adapter — declaring one would be inert
        # (pitfall #16), since the strategy re-derives the image from the raw
        # ``measured_kspace`` and never consumes the adapter output. Exempt the
        # kspace→image domain leg; spatial-dims is still cross-checked.
        if (
            self._is_equivariant_imaging_arm(config)
            and str(upstream.get("domain") or "").lower() in {"kspace", "k-space"}
            and str(required.get("domain") or "").lower() == "image"
        ):
            data_sd = upstream.get("spatial_dims")
            caps_sd = tuple(caps.spatial_dims or ())
            if data_sd is None or not caps_sd or data_sd in caps_sd:
                return HealthCheckResult(
                    passed=True,
                    check_name="data_model_compatibility",
                    message=(
                        "Equivariant Imaging strategy adjoints k-space to image "
                        "internally (A^* via _prepare_generator_inputs); the "
                        "kspace→image domain bridge is handled by the strategy "
                        "(a declared pre_model adapter would be inert) — "
                        "domain cross-check skipped."
                    ),
                    severity="info",
                    category="data_model_compatibility",
                )

        # Apply pre_model adapter chain if declared.
        adapters_cfg = getattr(config, "adapters", None)
        chain_names: list[str] = []
        if adapters_cfg is not None:
            chain_names = [
                step.name
                for step in getattr(adapters_cfg, "pre_model", [])
                if getattr(step, "enabled", True)
            ]

        if chain_names:
            chain_result = compose_chain(upstream, chain_names)
            if not chain_result.bridged:
                return HealthCheckResult(
                    passed=False,
                    check_name="data_model_compatibility",
                    message=(
                        f"adapters.pre_model chain {chain_names} is broken: {chain_result.error}"
                    ),
                    severity="error",
                    category="data_model_compatibility",
                    yaml_keys=["adapters.pre_model"],
                )
            upstream_after = chain_result.output_form
        else:
            upstream_after = upstream

        ok, mismatched = forms_match(produced=upstream_after, required=required)
        if ok:
            if chain_names:
                msg = (
                    f"data ⇆ model bridged by adapters.pre_model={chain_names}. "
                    f"Side-effects: {chain_result.side_effects or 'none'}."
                )
            else:
                msg = "data ⇆ model capabilities match (no adapter required)."
            return HealthCheckResult(
                passed=True,
                check_name="data_model_compatibility",
                message=msg,
                severity="info",
            )

        # Hard mismatch and no (or insufficient) adapter chain → error.
        candidates = candidate_adapters_for(
            upstream_form=upstream_after,
            required_form=required,
            hook="pre_model",
        )
        suggest = (
            f"Add adapters.pre_model: [{', '.join(candidates)}] (one of these "
            "single-step bridges fits)."
            if candidates
            else "No registered adapter bridges this gap; either register a new "
            "adapter or change the data/model so capabilities align."
        )
        return HealthCheckResult(
            passed=False,
            check_name="data_model_compatibility",
            message=(
                f"Data form {upstream_after} does not satisfy model "
                f"capabilities {required} (mismatched: {mismatched}). "
                f"Per CLAUDE.md #9 no silent rescue — declare an adapter "
                f"explicitly under `adapters.pre_model:`."
            ),
            severity="error",
            category="data_model_compatibility",
            yaml_keys=[
                "data.sampling.patch_size",
                "data.dataset_type",
                "model.model_type",
                "adapters.pre_model",
            ],
            fix_hint=suggest,
        )

    # ------------------------------------------------------------------
    # Phase 8: Transforms + Physics contracts
    # ------------------------------------------------------------------

    def check_sfc_scan_mode_matches_spatial_dims(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """Reject SFC scan_mode whose rank disagrees with data spatial_dims.

        ``model.model_kwargs.scan_mode`` is the linearizer the model
        uses to flatten the spatial grid into a 1-D Mamba sequence.
        ``hilbert_2d`` requires rank-2 input, ``hilbert_3d`` rank-3,
        ``raster_1d`` rank-1, etc. A mismatch causes the linearizer to
        crash at first batch with an unhelpful index error.
        """
        from spectramr.infrastructure.validation.spec_card import _derive_data_form

        data = getattr(config, "data", None)
        model = getattr(config, "model", None)
        if data is None or model is None:
            return HealthCheckResult(
                passed=True,
                check_name="sfc_scan_mode",
                message="data or model section absent; skipping.",
                severity="info",
            )
        mk = getattr(model, "model_kwargs", None) or {}
        if not isinstance(mk, dict):
            return HealthCheckResult(
                passed=True,
                check_name="sfc_scan_mode",
                message="model.model_kwargs not a dict; skipping.",
                severity="info",
            )
        scan_mode = mk.get("scan_mode")
        if not scan_mode:
            return HealthCheckResult(
                passed=True,
                check_name="sfc_scan_mode",
                message="No scan_mode declared; not an SFC-using model.",
                severity="info",
            )
        # The rank advertised by the scan_mode suffix.
        suffix_to_rank = {"_1d": 1, "_2d": 2, "_3d": 3}
        expected_rank = next(
            (r for s, r in suffix_to_rank.items() if scan_mode.endswith(s)),
            None,
        )
        if expected_rank is None:
            return HealthCheckResult(
                passed=True,
                check_name="sfc_scan_mode",
                message=f"scan_mode={scan_mode!r} has no _Nd suffix; skipping rank check.",
                severity="info",
            )
        # Rank from the model's spatial_shape kwarg (if declared) takes
        # precedence over data; fall back to data patch_size rank.
        spatial_shape = mk.get("spatial_shape")
        if spatial_shape and isinstance(spatial_shape, (list, tuple)):
            actual_rank = len(spatial_shape)
            source = "model.model_kwargs.spatial_shape"
        else:
            data_form = _derive_data_form(config)
            actual_rank = data_form.get("spatial_dims")
            source = "data.sampling.patch_size"
        if actual_rank is None:
            return HealthCheckResult(
                passed=True,
                check_name="sfc_scan_mode",
                message=f"Cannot infer spatial rank from {source}; skipping.",
                severity="info",
            )
        if actual_rank != expected_rank:
            return HealthCheckResult(
                passed=False,
                check_name="sfc_scan_mode",
                message=(
                    f"scan_mode={scan_mode!r} requires rank {expected_rank} "
                    f"but {source} declares rank {actual_rank}."
                ),
                severity="error",
                category="sfc_scan_mode",
                yaml_keys=["model.model_kwargs.scan_mode", source],
                fix_hint=(
                    f"Either set scan_mode to a rank-{actual_rank} variant "
                    f"(e.g. raster_{actual_rank}d, hilbert_{actual_rank}d) "
                    f"or change {source} to rank {expected_rank}."
                ),
            )
        return HealthCheckResult(
            passed=True,
            check_name="sfc_scan_mode",
            message=f"scan_mode={scan_mode!r} (rank {expected_rank}) matches {source} (rank {actual_rank}).",
            severity="info",
        )

    def check_data_consistency_requires_kspace(self, config: TrainingSettings) -> HealthCheckResult:
        """``physics.data_consistency.enabled=true`` requires k-space data.

        DC layers project the model output onto the measured k-space —
        if the data isn't k-space (or readily FFT-able from image), the
        DC step has no measured k-space to project against and the
        layer falls back to no-op.
        """
        from spectramr.infrastructure.validation.spec_card import _derive_data_form

        physics = getattr(config, "physics", None)
        if physics is None:
            return HealthCheckResult(
                passed=True,
                check_name="data_consistency_kspace",
                message="No physics section.",
                severity="info",
            )
        dc = getattr(physics, "data_consistency", None)
        if dc is None or not getattr(dc, "enabled", False):
            return HealthCheckResult(
                passed=True,
                check_name="data_consistency_kspace",
                message="DC disabled; not applicable.",
                severity="info",
            )
        data_form = _derive_data_form(config)
        domain = data_form.get("domain_inferred")
        # k-space, complex_image, and image-domain (with implicit FFT in DC)
        # are all valid. Pure pde_grid / mesh / latent are not.
        valid_domains = {"kspace", "complex_image", "image"}
        if domain in valid_domains:
            return HealthCheckResult(
                passed=True,
                check_name="data_consistency_kspace",
                message=f"DC enabled; data domain={domain!r} compatible.",
                severity="info",
            )
        return HealthCheckResult(
            passed=False,
            check_name="data_consistency_kspace",
            message=(
                f"physics.data_consistency.enabled=true but data domain is "
                f"{domain!r}. DC needs k-space (or image to FFT into k-space) "
                "to project against measurements."
            ),
            severity="error",
            category="physics_contract",
            yaml_keys=["physics.data_consistency.enabled", "data.dataset_type"],
            fix_hint=(
                "Either disable physics.data_consistency or switch the "
                "dataset to a k-space / image source."
            ),
        )

    def check_discriminator_requires_gan_losses(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """F15 (2026-05-21 smoke audit): when the YAML declares a
        discriminator (under ``model.discriminator_component`` or via the
        legacy ``model.discriminator`` block), ``losses.gan`` must also
        be present — otherwise ``_build_composite_gan`` doesn't build the
        discriminator optimizer and ``Trainer.execute_step`` crashes
        with ``step_configs[1] (name='discriminator') has optimizer=None``.

        This Tier-1 check catches the bug at audit time so the smoke
        wrapper's audit gate stops the experiment at submission, not
        runtime. See TODO/audit/smoke_audit_20260521.md §F11+§F15.
        """
        model = getattr(config, "model", None)
        if model is None:
            return HealthCheckResult(
                passed=True,
                check_name="discriminator_gan_losses",
                message="No model section.",
                severity="info",
            )
        # Detect whether a discriminator is declared. Two layouts:
        # (1) v6.0+: ``model.discriminator_component`` (a sub-block with a
        #     ``name`` field naming a registered discriminator), OR
        # (2) legacy: ``model.discriminator`` (a sub-block).
        disc_component = getattr(model, "discriminator_component", None)
        disc_legacy = getattr(model, "discriminator", None)
        has_discriminator = (disc_component is not None) or (disc_legacy is not None)
        if not has_discriminator:
            return HealthCheckResult(
                passed=True,
                check_name="discriminator_gan_losses",
                message="No discriminator declared.",
                severity="info",
            )

        # When a discriminator is declared, ``losses.gan.enable_adversarial``
        # must be truthy. Otherwise the GAN composite isn't built.
        losses = getattr(config, "losses", None)
        gan = getattr(losses, "gan", None) if losses is not None else None
        enable_adv = getattr(gan, "enable_adversarial", False) if gan is not None else False
        if enable_adv:
            return HealthCheckResult(
                passed=True,
                check_name="discriminator_gan_losses",
                message="Discriminator declared and losses.gan present.",
                severity="info",
            )

        return HealthCheckResult(
            passed=False,
            check_name="discriminator_gan_losses",
            message=(
                "Model declares a discriminator (via "
                "``model.discriminator_component`` or legacy "
                "``model.discriminator``) but ``losses.gan`` is missing "
                "or ``enable_adversarial`` is false. The discriminator "
                "step would have ``optimizer=None`` at runtime and "
                "Trainer.execute_step would crash."
            ),
            severity="error",
            category="discriminator_gan_losses",
            yaml_keys=[
                "model.discriminator_component",
                "losses.gan.enable_adversarial",
            ],
            fix_hint=(
                "Add a ``losses.gan:`` block with at minimum "
                "``enable_adversarial: true`` (and ``lambda_adv``, "
                "``gan_loss_type``, ``disc_updates``). See "
                "src/config/schemas/loss.py GANLossesConfig for "
                "the full field set."
            ),
        )

    def check_acceleration_schedule_steps_match_diffusion(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """Reject ``acceleration.schedule_steps`` ≠ ``training.diffusion.timesteps``.

        The acceleration schedule covers t∈[0, schedule_steps); the diffusion
        process runs t∈[0, training.diffusion.timesteps). When the two
        disagree, the schedule either truncates early (sticks at max for
        most of training) or never reaches its configured tail. Both
        ruin the curriculum. Found 3 instances on first survey of the
        repo (dummy_kspace_cold_diffusion ×2, experiment_134).
        """
        accel = getattr(config, "undersampling", None)
        training = getattr(config, "training", None)
        if accel is None or training is None:
            return HealthCheckResult(
                passed=True,
                check_name="acceleration_schedule_steps_match",
                message="acceleration or training section absent.",
                severity="info",
            )
        diff_cfg = getattr(training, "diffusion", None)
        if diff_cfg is None:
            return HealthCheckResult(
                passed=True,
                check_name="acceleration_schedule_steps_match",
                message="Not a diffusion paradigm.",
                severity="info",
            )
        if getattr(accel, "declares_no_acceleration", False):
            return HealthCheckResult(
                passed=True,
                check_name="acceleration_schedule_steps_match",
                message="undersampling declares no acceleration (base and max 1.0): no mask schedule.",
                severity="info",
            )
        # `num_timesteps` fallbacks removed: no schema carries that field on
        # either receiver (the diffusion spelling folds to `timesteps`, and
        # `AccelerationConfigSchema` has only `schedule_steps`), so both halves
        # were dead. Unlike the two sites this change repairs, these read the
        # canonical name FIRST and were therefore correct all along.
        diff_steps = getattr(diff_cfg, "timesteps", None)
        accel_steps = getattr(accel, "schedule_steps", None)
        if diff_steps is None or accel_steps is None:
            return HealthCheckResult(
                passed=True,
                check_name="acceleration_schedule_steps_match",
                message="Either diffusion.timesteps or acceleration.schedule_steps unset; skipping.",
                severity="info",
            )
        # Fire when AT LEAST ONE of the two fields is user-declared.
        # Rationale (audit anchor TODO/audit/smoke_audit_20260516.md F3):
        # The pre-2026-05-17 gate required BOTH to be user-set, which
        # suppressed exactly the failure mode that produced 60 runtime
        # ERROR-level hits in the 2026-05-16 smoke log — YAMLs that
        # declare ``training.diffusion.timesteps`` and leave
        # ``schedule_steps`` at the schema default of 10000. The runtime
        # then notices the mismatch and aborts. Hoisting the check to
        # audit-time per CLAUDE.md pitfall #10 (warnings/runtime gates
        # masquerading as silent regressions) costs no false positives
        # because: (a) non-diffusion paradigms have ``training.diffusion
        # is None`` and were already skipped above; (b) when neither
        # field is user-set there's nothing to validate.
        try:
            accel_set = getattr(accel, "model_fields_set", set())
            diff_set = getattr(diff_cfg, "model_fields_set", set())
        except Exception:
            accel_set = set()
            diff_set = set()
        schedule_user_set = "schedule_steps" in accel_set
        timesteps_user_set = "timesteps" in diff_set or "num_timesteps" in diff_set
        if not (schedule_user_set or timesteps_user_set):
            return HealthCheckResult(
                passed=True,
                check_name="acceleration_schedule_steps_match",
                message=(
                    "Neither schedule_steps nor diffusion.timesteps "
                    "user-declared; nothing to enforce."
                ),
                severity="info",
            )
        if diff_steps == accel_steps:
            return HealthCheckResult(
                passed=True,
                check_name="acceleration_schedule_steps_match",
                message=f"diffusion.timesteps={diff_steps} matches acceleration.schedule_steps.",
                severity="info",
            )
        return HealthCheckResult(
            passed=False,
            check_name="acceleration_schedule_steps_match",
            message=(
                f"training.diffusion.timesteps={diff_steps} but "
                f"acceleration.schedule_steps={accel_steps}. The schedule will "
                "either saturate at max for most timesteps (if accel < diff) "
                "or never reach its tail (if accel > diff)."
            ),
            severity="error",
            category="physics_contract",
            yaml_keys=[
                "training.diffusion.timesteps",
                "acceleration.schedule_steps",
            ],
            fix_hint=f"Set both to the same value (typically {diff_steps}).",
        )

    def check_acceleration_consistency(self, config: TrainingSettings) -> HealthCheckResult:
        """Sanity-check the ``acceleration:`` block.

        Detected real bugs:
          - base_acceleration > max_acceleration (range collapses to empty).
          - center_fraction outside the supported [0.01, 0.5] band → silently
            either samples too few low-frequency lines (under) or so many
            that "acceleration" loses meaning (over). The 0.01 floor (not 0.04)
            is intentional: high-acceleration arms legitimately use ~0.03.
        """
        accel = getattr(config, "undersampling", None)
        if accel is None:
            return HealthCheckResult(
                passed=True,
                check_name="acceleration_consistency",
                message="No acceleration section.",
                severity="info",
            )
        base = getattr(accel, "base_acceleration", None)
        max_ = getattr(accel, "max_acceleration", None)
        cf = getattr(accel, "center_fraction", None)
        problems: list[str] = []
        if base is not None and max_ is not None and base > max_:
            problems.append(f"base_acceleration={base} > max_acceleration={max_} (empty schedule)")
        if cf is not None and (cf < 0.01 or cf > 0.5):
            problems.append(f"center_fraction={cf} outside the supported [0.01, 0.5] band")
        if problems:
            return HealthCheckResult(
                passed=False,
                check_name="acceleration_consistency",
                message="; ".join(problems),
                severity="error",
                category="physics_contract",
                yaml_keys=[
                    "acceleration.base_acceleration",
                    "acceleration.max_acceleration",
                    "acceleration.center_fraction",
                ],
                fix_hint=(
                    "base_acceleration ≤ max_acceleration; "
                    "0.01 ≤ center_fraction ≤ 0.5 (high-R arms legitimately use "
                    "~0.03; 0.08 is typical at R≈4)."
                ),
            )
        return HealthCheckResult(
            passed=True,
            check_name="acceleration_consistency",
            message="acceleration block consistent.",
            severity="info",
        )

    def check_validation_cascade_levels_in_range(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """For diffusion YAMLs with ``schedule_type='step'``, every cascade
        level the strategy probes at validation MUST appear in
        ``acceleration.acceleration_range``.

        Anchor: experiment_11_kspace_cold_diffusion mosaic triage 2026-05-28.

        The validation cascade in
        ``infrastructure/training/strategies/diffusion.py`` uses
        :meth:`~spectramr.infrastructure.physics.sampling.KSpaceAccelerator.timestep_for_acceleration`
        to pick a timestep for each cascade level. Under ``step``, that
        method raises ``ValueError`` for off-grid values — the strategy
        catches the error and skips the level. That keeps the metrics
        honest but quietly drops columns the YAML's author thought they
        were getting.

        Catching the mismatch at audit time gives the user an actionable
        fix (extend ``acceleration_range`` or change ``schedule_type``)
        before the next training launch.

        The ladder is read from
        :func:`~spectramr.core.cascading_validation.resolve_cascade_levels` --
        the same call the strategy makes -- so this check sees whatever
        ``validation.cascade.levels`` declares. It previously held its own
        ``(2.0, 8.0, 32.0)`` beside a docstring conceding the two "should be
        updated in lockstep", which is the two-owner shape non-negotiable 17
        forbids: a checker that green-lights an ``acceleration_range`` against
        a ladder the strategy no longer runs reports *nothing*, and the symptom
        surfaces as a missing column rather than a failed audit.
        """
        accel = getattr(config, "undersampling", None)
        training = getattr(config, "training", None)
        if accel is None or training is None:
            return HealthCheckResult(
                passed=True,
                check_name="validation_cascade_levels_in_range",
                message="acceleration or training section absent.",
                severity="info",
            )
        diff_cfg = getattr(training, "diffusion", None)
        if diff_cfg is None:
            return HealthCheckResult(
                passed=True,
                check_name="validation_cascade_levels_in_range",
                message="Not a diffusion paradigm; cascade does not run.",
                severity="info",
            )
        schedule_type = getattr(accel, "schedule_type", None) or getattr(
            accel, "acceleration_schedule", None
        )
        # AccelerationSchedule is an enum — prefer ``.value`` so "step" doesn't
        # get spelled "AccelerationSchedule.STEP" by ``str(...)``.
        sched_str = getattr(schedule_type, "value", schedule_type)
        if str(sched_str or "").lower() != "step":
            return HealthCheckResult(
                passed=True,
                check_name="validation_cascade_levels_in_range",
                message=f"schedule_type={sched_str!r} has a closed-form inverse; cascade is honest.",
                severity="info",
            )
        accel_range = getattr(accel, "acceleration_range", None)
        if not isinstance(accel_range, list) or not accel_range:
            return HealthCheckResult(
                passed=True,
                check_name="validation_cascade_levels_in_range",
                message="schedule_type='step' without explicit acceleration_range — binary fallback applies.",
                severity="info",
            )
        from spectramr.core.cascading_validation import resolve_cascade_levels

        try:
            cascade_levels = resolve_cascade_levels(getattr(config, "validation", None))
        except ValueError as exc:
            # An illegal ladder is the schema's error to raise, and it does.
            # Reaching here means the config was built around the schema, so
            # SAY the ladder is unreadable rather than falling back to the
            # default and checking a ladder this run will never evaluate
            # (non-negotiable 3).
            return HealthCheckResult(
                passed=False,
                check_name="validation_cascade_levels_in_range",
                message=f"validation.cascade.levels is not a legal ladder: {exc}",
                severity="warning",
                category="physics_contract",
                yaml_keys=["validation.cascade.levels"],
            )
        missing = [
            r for r in cascade_levels if not any(abs(float(x) - r) < 1e-6 for x in accel_range)
        ]
        if not missing:
            return HealthCheckResult(
                passed=True,
                check_name="validation_cascade_levels_in_range",
                message=f"all cascade levels {list(cascade_levels)} present in acceleration_range.",
                severity="info",
            )
        return HealthCheckResult(
            passed=False,
            check_name="validation_cascade_levels_in_range",
            message=(
                f"schedule_type='step' but cascade levels {missing} are "
                f"not in acceleration.acceleration_range={accel_range}. "
                "The diffusion strategy will skip those validation levels "
                f"(see TODO/audit/smoke_audit_20260528.md), so "
                f"val_*_<R>x columns for {missing} will never appear."
            ),
            severity="warning",
            category="physics_contract",
            yaml_keys=[
                "acceleration.schedule_type",
                "acceleration.acceleration_range",
            ],
            fix_hint=(
                "Either extend acceleration_range to include "
                f"{missing}, or switch schedule_type to a continuous "
                "schedule (linear/power_law/exponential) where every "
                "cascade level has an honest timestep inverse."
            ),
        )

    def check_timesteps_vs_step_buckets(self, config: TrainingSettings) -> HealthCheckResult:
        """Warn when ``training.diffusion.timesteps`` heavily over-samples
        the discrete ``acceleration.acceleration_range`` buckets.

        Anchor: experiment_11_kspace_cold_diffusion mosaic triage 2026-05-28.

        For ``schedule_type=step`` with N buckets, each bucket spans
        ``timesteps/N`` distinct t values but produces the SAME mask
        within the bucket (because the step schedule snaps to the
        bucket-index acceleration). With 1000 timesteps and 7 buckets
        the model sees 143 different t values per bucket but only 7
        unique masks — the time embedding sees 1000 distinct codes
        while the mask only has 7 states. For cold diffusion (where
        the mask IS the corruption operator) this is mostly
        capacity-wasteful, not incorrect. Worth flagging at audit time
        because the user can either (a) reduce ``timesteps`` to ≈ N,
        or (b) switch to a continuous schedule so every timestep
        carries new information.
        """
        accel = getattr(config, "undersampling", None)
        training = getattr(config, "training", None)
        if accel is None or training is None:
            return HealthCheckResult(
                passed=True,
                check_name="timesteps_vs_step_buckets",
                message="acceleration or training section absent.",
                severity="info",
            )
        diff_cfg = getattr(training, "diffusion", None)
        if diff_cfg is None:
            return HealthCheckResult(
                passed=True,
                check_name="timesteps_vs_step_buckets",
                message="Not a diffusion paradigm.",
                severity="info",
            )
        schedule_type = getattr(accel, "schedule_type", None) or getattr(
            accel, "acceleration_schedule", None
        )
        sched_str = str(getattr(schedule_type, "value", schedule_type) or "").lower()
        if sched_str != "step":
            return HealthCheckResult(
                passed=True,
                check_name="timesteps_vs_step_buckets",
                message=f"schedule_type={sched_str!r} is continuous; every timestep carries new info.",
                severity="info",
            )
        accel_range = getattr(accel, "acceleration_range", None)
        if not isinstance(accel_range, list) or len(accel_range) < 2:
            return HealthCheckResult(
                passed=True,
                check_name="timesteps_vs_step_buckets",
                message="schedule_type='step' without an explicit range; binary fallback applies.",
                severity="info",
            )
        timesteps = getattr(diff_cfg, "timesteps", None)
        if timesteps is None:
            return HealthCheckResult(
                passed=True,
                check_name="timesteps_vs_step_buckets",
                message="diffusion.timesteps unset; skipping.",
                severity="info",
            )
        n_buckets = len(accel_range)
        ratio = timesteps / max(1, n_buckets)
        # >10× over-sampling is wasteful enough to flag; <2× is fine
        # (cold diffusion still benefits from time-embedding gradient flow).
        if ratio <= 10.0:
            return HealthCheckResult(
                passed=True,
                check_name="timesteps_vs_step_buckets",
                message=(
                    f"timesteps={timesteps} / {n_buckets} buckets = {ratio:.1f}× (acceptable)."
                ),
                severity="info",
            )
        return HealthCheckResult(
            passed=False,
            check_name="timesteps_vs_step_buckets",
            message=(
                f"diffusion.timesteps={timesteps} oversamples the "
                f"{n_buckets}-bucket step schedule by {ratio:.0f}× — "
                f"the model sees {timesteps} distinct time codes but the "
                f"mask only has {n_buckets} states (all timesteps in a "
                f"bucket share the same mask). Time-embedding capacity "
                f"is largely wasted."
            ),
            severity="warning",
            category="physics_contract",
            yaml_keys=[
                "training.diffusion.timesteps",
                "acceleration.schedule_type",
                "acceleration.acceleration_range",
            ],
            fix_hint=(
                f"Either set timesteps ≈ {n_buckets * 4} (a small "
                "multiple of the bucket count) or switch schedule_type "
                "to linear/power_law so every timestep gets a unique R."
            ),
        )

    def check_time_keys_within_num_timesteps(self, config: TrainingSettings) -> HealthCheckResult:
        """Reject diffusion configs whose t-valued keys exceed ``num_timesteps``.

        Anchor: experiment_11_kspace_cold_diffusion mosaic triage 2026-05-28.

        Three keys carry diffusion-timestep semantics (values in
        ``[0, T)``) but live OUTSIDE the ``training.diffusion`` block:

        * ``training.curriculum_start_timestep`` — initial cap on the
          training-time timestep sampler. ``current_max = min(T,
          start_t + iter * rate)`` (diffusion.py:432-435) — when
          ``start_t > T`` the curriculum saturates from iter 0.
        * ``training.sampling_steps`` and
          ``training.diffusion.sampling_steps`` — the reverse-diffusion
          inference stride. ``cold_diffusion.py:570`` sets
          ``scheduler.set_timesteps(sampling_steps)``; values above
          ``T`` walk past the schedule's terminal index and either
          crash or silently clip.
        * ``model.model_kwargs.timesteps`` — the time-embedding's
          ``max_timesteps`` (``kspace_cold_diffusion_generator.py:2369``).
          A mismatch with ``training.diffusion.timesteps`` collapses
          all sampled timesteps into a tiny embedding region.

        The audit asserts every one of these is consistent with the
        canonical ``training.diffusion.timesteps`` value.
        """
        training = getattr(config, "training", None)
        if training is None:
            return HealthCheckResult(
                passed=True,
                check_name="time_keys_within_num_timesteps",
                message="training section absent.",
                severity="info",
            )
        diff_cfg = getattr(training, "diffusion", None)
        if diff_cfg is None:
            return HealthCheckResult(
                passed=True,
                check_name="time_keys_within_num_timesteps",
                message="Not a diffusion paradigm.",
                severity="info",
            )
        T = getattr(diff_cfg, "timesteps", None)
        if T is None:
            return HealthCheckResult(
                passed=True,
                check_name="time_keys_within_num_timesteps",
                message="training.diffusion.timesteps unset; skipping.",
                severity="info",
            )

        problems: list[str] = []
        # 1. curriculum_start_timestep < T
        start_t = getattr(training, "curriculum_start_timestep", None)
        if start_t is not None and int(start_t) >= int(T):
            problems.append(
                f"training.curriculum_start_timestep={start_t} >= "
                f"training.diffusion.timesteps={T} — curriculum "
                "saturates at the first iteration."
            )

        # 2. training.sampling_steps (legacy top-level) <= T
        top_ss = getattr(training, "sampling_steps", None)
        if top_ss is not None and int(top_ss) > int(T):
            problems.append(
                f"training.sampling_steps={top_ss} > timesteps={T} — "
                "reverse-diffusion stride walks past terminal index."
            )

        # 3. training.diffusion.sampling_steps <= T
        diff_ss = getattr(diff_cfg, "sampling_steps", None)
        if diff_ss is not None and int(diff_ss) > int(T):
            problems.append(
                f"training.diffusion.sampling_steps={diff_ss} > "
                f"timesteps={T} — reverse-diffusion stride walks past "
                "terminal index."
            )

        # 4. model.model_kwargs.timesteps == T (time-embedding max_t)
        model = getattr(config, "model", None)
        if model is not None:
            mkw = dict(getattr(model, "model_kwargs", {}) or {})
            model_T = mkw.get("timesteps")
            if model_T is not None and int(model_T) != int(T):
                problems.append(
                    f"model.model_kwargs.timesteps={model_T} != "
                    f"training.diffusion.timesteps={T} — the time-"
                    "embedding's max_timesteps mismatch collapses all "
                    "actually-used timesteps into a tiny embedding band."
                )

        if not problems:
            return HealthCheckResult(
                passed=True,
                check_name="time_keys_within_num_timesteps",
                message=(f"all t-valued keys consistent with training.diffusion.timesteps={T}."),
                severity="info",
            )
        return HealthCheckResult(
            passed=False,
            check_name="time_keys_within_num_timesteps",
            message=" / ".join(problems),
            severity="error",
            category="physics_contract",
            yaml_keys=[
                "training.diffusion.timesteps",
                "training.curriculum_start_timestep",
                "training.sampling_steps",
                "training.diffusion.sampling_steps",
                "model.model_kwargs.timesteps",
            ],
            fix_hint=(
                f"Make every t-valued key respect "
                f"training.diffusion.timesteps={T}: "
                f"curriculum_start_timestep ≤ {max(0, int(T) - 1)}, "
                f"sampling_steps ≤ {T}, model.model_kwargs.timesteps == {T}."
            ),
        )

    def check_metric_transforms_agree(self, config: TrainingSettings) -> HealthCheckResult:
        """``train_<m>`` and ``val_<m>`` must measure the same thing to be compared.

        The training-metric path reads ``metrics.transform`` and the validation
        path reads ``validation.scoring.output_transform``. When they differ the
        two numbers share a name stem and measure different quantities -- on 69
        cohort arms, ``ifft_sense_adjoint`` in normalised log-compressed units
        against ``ifft_magnitude`` in denormalised physical ones -- so the gap
        between them is a pipeline artefact that reads as a generalisation gap.

        Advisory: making them agree changes every reported ``train_`` metric on
        every affected arm, which is an owner decision rather than a repair.
        """
        check_name = "metric_transforms_agree"
        train_t = getattr(getattr(config, "metrics", None), "transform", None)
        scoring = getattr(getattr(config, "validation", None), "scoring", None)
        val_t = getattr(scoring, "output_transform", None) if scoring else None
        if train_t is None or val_t is None or str(train_t) == str(val_t):
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message=(
                    f"training and validation metrics share one transform "
                    f"({train_t!r})."
                    if train_t is not None
                    else "no metric transform declared on either path."
                ),
                severity="info",
            )
        return HealthCheckResult(
            passed=True,
            check_name=check_name,
            message=(
                f"metrics.transform={train_t!r} but "
                f"validation.scoring.output_transform={val_t!r}, so train_<m> and "
                f"val_<m> are different measurements under one name. Their "
                f"difference is a pipeline artefact, not a generalisation gap."
            ),
            severity="info",
            category="metrics",
            yaml_keys=["metrics.transform", "validation.scoring.output_transform"],
            fix_hint=(
                "Declare the same transform on both paths, or read the two "
                "series separately and never as a train/val gap."
            ),
        )

    def check_kspace_scale_domain_is_declared(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """An undeclared scale domain is a silent choice of normalisation.

        ``KSpaceNormalizationSpec._read_declared_knobs`` already refuses a
        ``processing`` block that omits this knob, with a message naming #572.
        That refusal is unreachable in production: training hands it
        ``config.data`` and inference hands it ``config.model_dump()``, and
        pydantic has filled the default in both, so the only caller it can fire
        on is a raw YAML mapping that nothing passes. The check therefore has to
        ask the question here, where ``model_fields_set`` still records whether
        the author typed it.

        Advisory rather than blocking: 109 of the 185 configs that enable
        normalisation omit it, so raising would be a corpus migration. The
        polarity matches the workflow block -- absent is advisory, wrong is
        hard.
        """
        check_name = "kspace_scale_domain_is_declared"
        processing = getattr(getattr(config, "data", None), "processing", None)
        if processing is None or not getattr(processing, "enable_kspace_normalization", False):
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message="k-space normalization is off; the scale domain does not apply.",
                severity="info",
            )
        declared = getattr(processing, "model_fields_set", set())
        resolved = getattr(processing, "kspace_scale_domain", None)
        if "kspace_scale_domain" in declared:
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message=f"data.processing.kspace_scale_domain declared as {resolved!r}.",
                severity="info",
            )
        return HealthCheckResult(
            passed=True,
            check_name=check_name,
            message=(
                f"data.processing enables k-space normalization but declares no "
                f"kspace_scale_domain, so it resolves to the schema default "
                f"{resolved!r}. 'image' is the Parseval-compliant choice when "
                f"losses and metrics are graded in the image domain; an arm that "
                f"takes the default while its siblings declare 'image' is scored "
                f"on differently normalised predictions."
            ),
            severity="info",
            category="schema",
            yaml_keys=["data.processing.kspace_scale_domain"],
            fix_hint=(
                "Declare data.processing.kspace_scale_domain explicitly. Match "
                "the arm's declared metadata.baseline."
            ),
        )

    def check_dc_knobs_inert_by_method(self, config: TrainingSettings) -> HealthCheckResult:
        """DC knobs this arm declares that its resolved ``dc_method`` cannot read.

        Closes the gap :meth:`check_declared_model_kwargs_are_read` names and
        declines: that check is static, so it answers only for parameters a
        constructor never mentions, and explicitly cannot see a knob that is
        inert **by method**. ``inert_knobs.py`` coins the term using this exact
        example and records that nothing reports it. This is the report.

        ``dc_weight`` is not a universal blend fraction. It is ``lambda_init``
        for :class:`SoftDataConsistency`, ``beta`` for
        :class:`NoiseAdaptiveDataConsistency`, ``hf_lambda`` for
        :class:`TargetAwareFSDC` and ``weight`` for
        :class:`SimpleDataConsistency`. :class:`HardDataConsistency` takes **no
        weight argument**: its blend is ``(1 - m) * recon + m * obs``, which is
        weight 1.0 by construction. The reverse path agrees -- ``p_sample`` and
        ``_apply_observed_dc`` branch on ``dc_method == "hard"`` before reaching
        the ``dc_weight`` term -- so a declared weight is inert under ``hard`` at
        training *and* at sampling.

        The fix is therefore to **delete the key**, never to make hard DC honour
        it; honouring it would silently convert hard DC into soft DC.

        Only values differing from the schema default are reported. Every
        declared Pydantic field is emitted by ``model_dump``, so a presence test
        would be a tautology, and a defaulted value is not a choice the author
        made.

        **Advisory, and the corpus IS counted.** Census 2026-08-26 over all 647
        arms under ``experiments/inprogress`` (647 loaded, 0 unloadable):
        **54 arms (8.3%) declare a DC knob their method cannot read** -- all 54
        are ``dc_weight``, across 2 cohorts (``kspace_filling`` 53,
        ``cold_diffusion`` 1), at values 0.5 and 0.3. 133 arms resolve to
        ``dc_method: hard``, so the other 79 declare the default 1.0 and are
        correctly silent. No arm declares an inert noise knob: the two that set
        non-default noise levels resolve to a method that falls through to
        ``SimpleDataConsistency``, which accepts them.

        A ``warning`` exits 2 under the strict smoke wrapper (pitfall #10), so
        raising severity now would take down 54 arms for a defect none of them
        introduced. Ratchet to ``warning`` once that corpus reaches zero.
        """
        check_name = "dc_knobs_inert_by_method"
        findings = inert_dc_knobs(config)
        if not findings:
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message="no declared DC knob is inert under the resolved dc_method.",
                severity="info",
            )

        resolved = resolve_effective_dc(config)
        detail = " / ".join(
            f"{knob}={value!r} is inert ({reason})" for knob, value, reason in findings
        )
        return HealthCheckResult(
            passed=True,  # advisory while the 54-arm corpus is non-zero
            check_name=check_name,
            message=(
                f"{detail}. The value is stamped into provenance and reads as a "
                f"chosen experimental variable, but changes nothing."
            ),
            severity="info",
            category="physics_contract",
            yaml_keys=[f"physics.data_consistency.{k}" for k, _, _ in findings],
            fix_hint=(
                f"Delete the key. Do NOT expect dc_method={resolved.method!r} to "
                f"apply it -- hard DC replaces the acquired bins rather than "
                f"blending toward them, so weight 1.0 is structural. Use "
                f"dc_method: soft (or noise_adaptive / target_aware_fsdc) if a "
                f"tunable trust temperature is what the arm actually wants."
            ),
        )

    def check_dc_method_physics_consistency(self, config: TrainingSettings) -> HealthCheckResult:
        """Cross-check ``model.model_kwargs.dc_method`` against ``physics.data_consistency``.

        Anchor: experiment_11_kspace_cold_diffusion mosaic triage 2026-05-28.

        Three states matter to the user:

        1. ``physics.data_consistency.enabled=true`` AND model's
           ``dc_method`` is one of the disable sentinels
           (``null`` / ``""`` / ``"none"`` / ``"off"`` / ``"disabled"``)
           — the user toggled both knobs in opposite directions; one of
           them silently wins (after the 2026-05-28 fix the disable
           wins). Either intent is reasonable, but the YAML reads as
           contradictory and the warning makes the resolution visible.

        2. ``physics.data_consistency.enabled=false`` AND model's
           ``dc_method`` is set to something other than a disable
           sentinel — same contradiction, opposite direction. The
           model still applies DC because the SSOT reconciliation in
           ``ModelBuilder.build_generator`` overwrites the model-side
           ``dc_method`` only when ``physics.data_consistency.enabled``
           is true; otherwise the model-side value takes effect.

        3. Both agree — silent pass.

        The SSOT reconciliation lives in
        ``infrastructure/training/builders/model_builder.py:150-189``.
        """
        model = getattr(config, "model", None)
        physics = getattr(config, "physics", None)
        if model is None or physics is None:
            return HealthCheckResult(
                passed=True,
                check_name="dc_method_physics_consistency",
                message="model or physics section absent.",
                severity="info",
            )
        dc = getattr(physics, "data_consistency", None)
        if dc is None:
            return HealthCheckResult(
                passed=True,
                check_name="dc_method_physics_consistency",
                message="physics.data_consistency absent.",
                severity="info",
            )
        phys_enabled = bool(getattr(dc, "enabled", False))
        model_kwargs = dict(getattr(model, "model_kwargs", {}) or {})
        model_dc_method = model_kwargs.get("dc_method", "__unset__")

        _DISABLE_SENTINELS = (None, "", "none", "off", "disabled")
        model_dc_disabled = model_dc_method in _DISABLE_SENTINELS
        model_dc_explicit_method = model_dc_method != "__unset__" and not model_dc_disabled

        if phys_enabled and model_dc_disabled and model_dc_method != "__unset__":
            return HealthCheckResult(
                passed=False,
                check_name="dc_method_physics_consistency",
                message=(
                    f"physics.data_consistency.enabled=true but "
                    f"model.model_kwargs.dc_method={model_dc_method!r} "
                    "is a disable sentinel. The 2026-05-28 dispatcher "
                    "fix in KSpaceColdDiffusionGenerator honours the "
                    "explicit disable, so DC will NOT be applied "
                    "despite physics.data_consistency claiming it is."
                ),
                severity="warning",
                category="config_consistency",
                yaml_keys=[
                    "model.model_kwargs.dc_method",
                    "physics.data_consistency.enabled",
                    "physics.data_consistency.method",
                ],
                fix_hint=(
                    "Pick ONE source of truth. To use the physics block, "
                    "either delete model_kwargs.dc_method or set it to "
                    "match physics.data_consistency.method. To disable DC, "
                    "delete the physics.data_consistency block (or set "
                    "physics.data_consistency.enabled=false)."
                ),
            )
        if (not phys_enabled) and model_dc_explicit_method:
            return HealthCheckResult(
                passed=False,
                check_name="dc_method_physics_consistency",
                message=(
                    f"physics.data_consistency.enabled=false but "
                    f"model.model_kwargs.dc_method={model_dc_method!r} "
                    "(non-disable). The model will apply DC despite the "
                    "physics block disabling it."
                ),
                severity="warning",
                category="config_consistency",
                yaml_keys=[
                    "model.model_kwargs.dc_method",
                    "physics.data_consistency.enabled",
                ],
                fix_hint=(
                    "Pick ONE source of truth. To keep DC, set "
                    "physics.data_consistency.enabled=true (the SSOT "
                    "reconciliation will then mirror its method/weight "
                    "into the model). To disable DC, set "
                    "model_kwargs.dc_method=null."
                ),
            )
        return HealthCheckResult(
            passed=True,
            check_name="dc_method_physics_consistency",
            message="model.dc_method and physics.data_consistency agree.",
            severity="info",
        )

    def check_validation_split_redundancy(self, config: TrainingSettings) -> HealthCheckResult:
        """Flag ``validation.split`` when an authoritative source is also set.

        Anchor: experiment_11_kspace_cold_diffusion mosaic triage 2026-05-28.

        The validation set composition has three competing knobs:
          * ``data.split_strategy: manifest`` + ``data.validation_index_path``
            (manifest is authoritative)
          * ``data.validation_split`` (random fraction — read by
            ``data/datasets/factory.py`` and ``data/metadata/index_builder.py``)
          * ``validation.split`` (read by NOTHING in src/spectramr — a dead
            knob; CLAUDE.md pitfall #15)

        Flagging the conflict at audit time prevents the YAML author
        from believing they've set the val fraction when the runtime
        actually loaded a manifest.
        """
        data = getattr(config, "data", None)
        val = getattr(config, "validation", None)
        if data is None or val is None:
            return HealthCheckResult(
                passed=True,
                check_name="validation_split_redundancy",
                message="data or validation section absent.",
                severity="info",
            )
        val_split_set = "split" in getattr(val, "model_fields_set", set())
        split_strategy = data.split.type
        validation_index_path = data.source.validation_index_path
        if not val_split_set:
            return HealthCheckResult(
                passed=True,
                check_name="validation_split_redundancy",
                message="validation.split not user-declared.",
                severity="info",
            )
        problems = []
        if validation_index_path is not None:
            problems.append(
                f"validation_index_path={validation_index_path!r} is "
                "authoritative — validation.split is silently ignored."
            )
        if (
            split_strategy is not None
            and str(getattr(split_strategy, "value", split_strategy)).lower() == "manifest"
        ):
            problems.append(
                "data.split.type='manifest' loads the val set from the "
                "manifest — validation.split is silently ignored."
            )
        # Dead-knob warning even when nothing else is set (validation.split
        # is read by no code in src/spectramr as of 2026-05-28).
        problems.append(
            "validation.split has no reader in src/spectramr (CLAUDE.md #15: "
            "exposed knob with no consumer). Use "
            "data.split.validation_fraction for the random-split case."
        )
        return HealthCheckResult(
            passed=False,
            check_name="validation_split_redundancy",
            message=" / ".join(problems),
            severity="warning",
            category="config_consistency",
            yaml_keys=[
                "validation.split",
                "data.split.type",
                "data.validation_index_path",
                "data.split.validation_fraction",
            ],
            fix_hint=(
                "Remove validation.split. For a manifest-driven split "
                "(experiment_11), keep data.validation_index_path. For "
                "a random split, set data.split.validation_fraction instead."
            ),
        )

    def check_declared_keys_are_not_discarded(self, config: TrainingSettings) -> HealthCheckResult:
        """Surface keys the YAML declares that the schema silently discarded.

        A block with ``extra="ignore"`` accepts an unknown key, drops it before
        the model exists, and says nothing. The YAML still shows it; the run
        never sees it. That is issue #550's mechanism, and it is invisible to
        every other check here because the resolved config is *correct* -- the
        value simply is not in it. Measured across ``experiments/``: 26 phantom
        keys under ``logging:`` across **1,154 declarations** (419 arms set
        ``project_name``, 417 set ``enable_wandb``, neither exists) and 125 more
        under ``undersampling:`` -- issues #675 and #681, which this closes.

        Some of them look load-bearing: ``data.input_hr_dir`` /
        ``data.input_lr_dir`` are declared and discarded on real arms, so the
        run reads whatever the schema's own fields resolve to instead.

        **Advisory, deliberately.** The corpus has ~1,279 of these, so a
        ``warning`` would exit 2 under the strict smoke wrapper (pitfall #10)
        and fail hundreds of arms at once for something that changes no
        behaviour. Same polarity as ``check_workflow_declared``: report it,
        ratchet to error once the corpus is drained. Issue #674's leaf-name
        blindness is unrelated -- this reads the ledger's dotted paths.

        Reads ``ExecutionLedger.current()``, which ``spectramr audit`` arms before
        resolving the config. When no ledger is active the answer is **"not
        measured"**, never "none found" -- a check that cannot tell those apart
        is the silence this one exists to break.
        """
        try:
            from spectramr.core.execution_ledger import (
                ExecutionLedger,
                SubstitutionClass,
            )
        except Exception:  # pragma: no cover - core is always importable
            return HealthCheckResult(
                passed=True,
                check_name="declared_keys_are_not_discarded",
                message="execution ledger unavailable; discarded keys NOT measured.",
                severity="info",
            )

        ledger = ExecutionLedger.current()
        if ledger is None:
            return HealthCheckResult(
                passed=True,
                check_name="declared_keys_are_not_discarded",
                message=(
                    "no execution ledger armed for this load, so discarded keys "
                    "were NOT measured. This is not a clean result. Arm one with "
                    "ExecutionLedger.begin_run() before TrainingSettings.from_yaml()."
                ),
                severity="info",
            )

        dropped = [
            sub.path
            for sub in ledger.substitutions
            if sub.class_id is SubstitutionClass.EXTRA_IGNORE_DROPPED
        ]
        if not dropped:
            return HealthCheckResult(
                passed=True,
                check_name="declared_keys_are_not_discarded",
                message="every declared key reached the resolved config.",
                severity="info",
            )

        shown = ", ".join(sorted(dropped)[:8])
        more = f" (+{len(dropped) - 8} more)" if len(dropped) > 8 else ""
        return HealthCheckResult(
            passed=True,  # advisory until the corpus is drained; see docstring
            check_name="declared_keys_are_not_discarded",
            message=(
                f"{len(dropped)} declared key(s) were discarded by an "
                f"extra='ignore' block and the run never sees them: {shown}{more}"
            ),
            severity="info",
            category="schema",
            yaml_keys=sorted(dropped),
            fix_hint=(
                "Delete the key, or declare it on the schema block if it is "
                "meant to be read. The YAML currently advertises a knob that "
                "does nothing (pitfall #15)."
            ),
        )

    def check_component_kwargs_reach_constructor(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """Declared component kwargs that never reached a constructor.

        A kwarg that does not reach the constructor changes what the run
        computes while every artifact still reads clean — `sobolev_order: 1` sat
        declared-and-dead in 56 arms on exactly this (#560, #615).

        **Scope: models, not losses.** The loss builder RAISES on an unroutable
        kwarg, so a loss violation never reaches an audit. This reads the two
        classes that legitimately cannot raise:

        ``DROPPED_UNCONSUMED_KWARG``
            the signature filter removed the key; injecting ``model_kwargs``
            into a constructor that never declared them is a supported
            flexibility path, so dropping is correct and raising would break
            the corpus.
        ``EXTRA_ALLOW_UNTYPED``
            a ``**kwargs`` constructor accepted the key and nothing proves it is
            read (#878).

        **Advisory, deliberately, and for a different reason than the sibling.**
        ``check_declared_keys_are_not_discarded`` is advisory because its corpus
        is ~1,279 and counted. This one is advisory because the model-kwargs
        corpus is **uncounted**: a ``warning`` exits 2 under the strict smoke
        wrapper (pitfall #10), and nobody has measured how many arms it would
        take down. Once someone censuses it — as was done for losses, which came
        back 0 and so could raise — this should ratchet on that evidence rather
        than on symmetry.

        Reads ``ExecutionLedger.current()``. When no ledger is active the answer
        is **"not measured"**, never "none found".
        """
        check_name = "component_kwargs_reach_constructor"
        try:
            from spectramr.core.execution_ledger import (
                ExecutionLedger,
                SubstitutionClass,
            )
        except Exception:  # pragma: no cover - core is always importable
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message="execution ledger unavailable; component kwargs NOT measured.",
                severity="info",
            )

        ledger = ExecutionLedger.current()
        if ledger is None:
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message=(
                    "no execution ledger armed for this load, so component "
                    "kwargs were NOT measured. This is not a clean result. Arm "
                    "one with ExecutionLedger.begin_run() before building."
                ),
                severity="info",
            )

        watched = {
            SubstitutionClass.DROPPED_UNCONSUMED_KWARG,
            SubstitutionClass.EXTRA_ALLOW_UNTYPED,
        }
        hits = [
            sub
            for sub in ledger.substitutions
            if sub.class_id in watched and sub.stage == "model_build"
        ]
        if not hits:
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message="every declared component kwarg reached its constructor.",
                severity="info",
            )

        shown = ", ".join(sorted(f"{s.path} -> {s.consumer or '?'}" for s in hits)[:6])
        more = f" (+{len(hits) - 6} more)" if len(hits) > 6 else ""
        return HealthCheckResult(
            passed=True,  # advisory until the model corpus is counted
            check_name=check_name,
            message=(
                f"{len(hits)} declared component kwarg(s) did not verifiably "
                f"reach a constructor: {shown}{more}"
            ),
            severity="info",
            category="schema",
            yaml_keys=sorted({s.path for s in hits}),
            fix_hint=(
                "Delete the key, or give the component a real parameter for it. "
                "A kwarg the constructor never names changes nothing but looks "
                "like it does (pitfall #15/#16)."
            ),
        )

    def check_declared_model_kwargs_are_read(self, config: TrainingSettings) -> HealthCheckResult:
        """Declared ``model_kwargs`` the resolved model provably never reads.

        The sibling ``check_component_kwargs_reach_constructor`` asks whether a
        declared kwarg **arrived** at the constructor. This asks the next
        question: having arrived, was it ever **read**? A parameter named in
        ``__init__``'s signature, documented in its docstring and then never
        referenced in the body arrives perfectly -- the ledger records a clean
        delivery -- and changes nothing. Measured on
        ``KSpaceColdDiffusionGenerator``: flipping ``activation`` or
        ``use_complex_conv`` leaves the module tree, the parameter count and the
        forward output bit-identical, and an invalid value does not raise
        because the validating resolver is never called with it.

        Arm-scoped on purpose. A dead parameter nobody sets is untidiness; a
        dead parameter an *experiment declares* is a false controlled variable --
        it reads as a knob the author chose, it is stamped into provenance, and
        in an ablation cohort it looks like a held-fixed axis. Only the
        intersection of "unread by this model" and "written by this arm" is
        reported, so every hit is something the author typed and expected to
        matter (non-negotiable 8, pitfall #15/#16).

        **Advisory, and the corpus IS counted** -- unlike its sibling, which is
        advisory because nobody had measured. Census 2026-08-20 over
        ``experiments/inprogress`` (647 arms with a resolvable ``model_type`` and
        a ``model_kwargs`` block): **90 arms (13.9%) declare at least one
        provably-unread knob** -- ``activation`` (82), ``use_complex_conv`` (82),
        ``img_size`` (5), ``features`` (1), ``bottleneck_only`` (1). **85** of
        those, across 9 cohorts, are what this check actually reports; the
        ``img_size`` 5 are suppressed by :data:`~spectramr.infrastructure.validation.inert_knobs.DELIBERATELY_UNREAD`. A
        ``warning`` exits 2 under the strict smoke wrapper (pitfall #10), so
        raising severity now would take down 90 arms for a defect none of them
        introduced. Ratchet to ``warning`` once that corpus reaches zero -- the
        fix is to give the constructor a real parameter or delete the key, not
        to silence the check.

        Static and conservative by construction: it answers only for parameters
        named in the signature, and declines to answer at all for a constructor
        that reaches for ``locals()``/``vars()``/``eval``. It therefore cannot
        see a knob that is inert *by method* (``dc_weight`` under
        ``dc_method: hard``, which replaces rather than blends) or *by branch*
        (``reflect_padding_bottleneck_layers`` under a backbone whose
        construction path never reads it). Those are properties of a
        configuration rather than of a class. The first is now covered by the
        sibling :meth:`check_dc_knobs_inert_by_method` (#1525); the second is
        still covered by nothing. A pass here means "no provably-unread declared
        knob", never "every declared knob matters".
        """
        check_name = "declared_model_kwargs_are_read"

        model_type = getattr(config.model, "model_type", None)
        declared = getattr(config.model, "model_kwargs", None) or {}
        if not model_type or not isinstance(declared, dict) or not declared:
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message="no model_kwargs declared; nothing to read.",
                severity="info",
            )

        try:
            from spectramr.infrastructure.validation.inert_knobs import (
                declared_knobs_out_of_scope,
                find_inert_declared_knobs,
            )
            from spectramr.models.init_registry import populate_model_registry
            from spectramr.models.registry import MODEL_REGISTRY

            if not MODEL_REGISTRY:
                populate_model_registry()
        except Exception as e:  # registry/import problems -> not measured
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message=f"model registry unavailable ({e}); knobs NOT measured.",
                severity="info",
            )

        entry = MODEL_REGISTRY.get(str(model_type))
        cls = entry.get("class") if isinstance(entry, dict) else None
        if cls is None:
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message=(
                    f"model_type={model_type!r} does not resolve to a class in "
                    "the decorator registry, so declared knobs were NOT "
                    "measured. This is not a clean result."
                ),
                severity="info",
            )

        inert = find_inert_declared_knobs(str(model_type), declared, cls)
        out_of_scope = declared_knobs_out_of_scope(declared, cls)
        if not inert:
            # State the scope, not just the verdict. This detector enumerates
            # named __init__ parameters only, so a key absorbed by **kwargs is
            # never a candidate -- 85.8 % of this corpus's declared keys on the
            # kspace_filling cohort. "All N are read" over that population reads
            # as a clean bill of health for knobs nothing looked at.
            measured = len(declared) - len(out_of_scope)
            scope = (
                f" {len(out_of_scope)} further key(s) are absorbed by **kwargs "
                f"and are NOT measured by this check; the package-wide reader "
                f"census (check_model_kwargs_are_read) is what covers those."
                if out_of_scope
                else ""
            )
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message=(
                    f"{measured} of {len(declared)} declared model_kwargs are in "
                    f"scope here and all are read by {cls.__name__} (or "
                    f"allowlisted as deliberate).{scope}"
                ),
                severity="info",
            )

        shown = ", ".join(f"{k.key}={k.declared_value!r}" for k in inert[:6])
        more = f" (+{len(inert) - 6} more)" if len(inert) > 6 else ""
        return HealthCheckResult(
            passed=True,  # advisory: 90/647 inprogress arms, censused 2026-08-20
            check_name=check_name,
            message=(
                f"{len(inert)} declared model_kwarg(s) are never read by "
                f"{cls.__name__}, so this arm's value cannot affect the run "
                f"while still being stamped into provenance: {shown}{more}"
            ),
            severity="info",
            category="schema",
            yaml_keys=[k.yaml_path for k in inert],
            fix_hint=(
                "Give the constructor a real parameter for it (forward it to "
                "the component that owns the behaviour), or delete the key. If "
                "it is kept deliberately for factory compatibility, add it to "
                "inert_knobs.DELIBERATELY_UNREAD with the reason."
            ),
        )

    def check_m4raw_nex_target_mode_declared(self, config: TrainingSettings) -> HealthCheckResult:
        """An ``m4raw`` arm should declare ``data.target_mode`` explicitly.

        M4Raw stores several **phase-incoherent** repetitions per anatomy --
        separate acquisitions with independent global phase drift. The schema
        default, ``complex_mean``, plain-averages them in complex k-space, which
        *cancels* signal instead of averaging it: the resulting "high-SNR
        reference" has SNR **below a single repetition** and scrambled phase.
        ``.claude/rules/data.md`` records where that led -- k-space cold
        diffusion learned to predict near-real k-space and produced a 180
        degree centro-symmetric doubled brain. ``phase_aligned_mean`` aligns each
        repetition's global phase to rep 0 first, giving the intended sqrt(N).

        **Advisory, and deliberately so.** The default is wrong *for M4Raw* but
        right for every other dataset, so it cannot simply be flipped. Measured
        2026-08-03 over ``git ls-files experiments`` (110 arms declare
        ``dataset_type: m4raw``; 107 resolve, 3 are unloadable and so never reach
        any check): **63 pass, 44 would newly fail**, 5 of them outside
        ``experiments/inprogress/``. A warning is not the softer option -- the
        smoke wrapper runs the audit ``--strict``, where warnings exit 2, so
        ``warning`` and ``error`` have the same blast radius. Ratcheting this to
        an error is an owner decision once those 44 are triaged (issue #694), the
        same posture ``check_workflow_declared`` takes for an absent
        ``workflow:`` block.

        **The vocabulary has one owner: the schema.** This check names only
        ``complex_mean`` (see ``_INCOHERENT_NEX_TARGET_MODE``) and treats every
        other member of the Literal as coherent. It used to enumerate the good
        modes instead, which silently went stale when ``r2r`` was added.

        **Absence is the finding, not the resolved value.** ``target_mode`` is a
        declared field with a default, so ``config.data.target_mode`` reads
        ``complex_mean`` whether the arm chose it or said nothing at all. Only
        ``model_fields_set`` separates a default nobody chose from an affirmative
        choice -- and those are different findings, so this reports them
        differently rather than collapsing both into one message.
        """
        check_name = "m4raw_nex_target_mode_declared"
        data = getattr(config, "data", None)
        if data is None or getattr(data, "dataset_type", None) != "m4raw":
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message="n/a: not an m4raw arm (NEX averaging is m4raw-only).",
                severity="info",
                category="data",
            )

        declared = "target_mode" in data.model_fields_set
        mode = data.target_mode
        if declared and mode != _INCOHERENT_NEX_TARGET_MODE:
            # Phrased as "anything but complex_mean" on purpose -- see
            # ``_INCOHERENT_NEX_TARGET_MODE`` for why an allowlist here is a
            # second owner of the schema Literal, and for the r2r arm it
            # mislabelled while it was one.
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message=f"data.target_mode: {mode} (coherent NEX target).",
                severity="info",
                category="data",
                yaml_keys=["data.target_mode"],
            )

        detail = (
            # ``mode``, never a literal name -- printing a constant here is how
            # an r2r arm came to be told it had declared complex_mean. With the
            # guard above admitting every mode but ``complex_mean`` this branch
            # can no longer be reached by anything else, so the f-string is
            # insurance against that guard being narrowed again rather than a
            # difference any test can observe today (measured: swapping it back
            # for the constant turns nothing red).
            f"declares data.target_mode: {mode} explicitly"
            if declared
            else ("does not declare data.target_mode, so it takes the schema default complex_mean")
        )
        return HealthCheckResult(
            passed=False,
            check_name=check_name,
            message=(
                f"m4raw arm {detail}. Complex-averaging phase-incoherent "
                f"repetitions cancels signal: the NEX target ends up with LOWER "
                f"SNR than one repetition and corrupted phase, so validation "
                f"PSNR/SSIM grade against a degraded reference."
            ),
            severity="info",
            category="data",
            yaml_keys=["data.target_mode"],
            fix_hint=(
                "Set data.target_mode: phase_aligned_mean. Use complex_mean only "
                "for a deliberate legacy comparison, and say so in the arm's "
                "metadata -- see .claude/rules/data.md 'M4Raw handling'."
            ),
        )

    def check_slice_level_records_queue_shape(self, config: TrainingSettings) -> HealthCheckResult:
        """``data.slice_level_records`` needs a depth-1 patch and one sample per record.

        The slice-level M4Raw route (#1757) indexes one record per (group,
        slice) and yields a depth-1 subject per record. Two config facts then
        decide whether the queue serves every slice exactly once per epoch:

        * ``sampling.patch_size`` depth must be 1. A deeper patch cannot be
          drawn from a depth-1 subject: at build time the queue's patch filter
          drops every record and the zero-batch guard raises. This reports the
          same failure before any data is opened.
        * ``samples_per_volume`` must be 1 unless the train sampler is ``full``
          (the no-queue bypass, where the knob is unread). ``tio.Queue`` draws
          that many patches from EVERY record per epoch; with a full-slice
          patch on M4Raw's 256 x 256 matrix they are identical copies of one
          slice, and a smaller patch is a k-space crop rather than an
          augmentation (``docs/troubleshooting.rst``). Either way the epoch
          double-samples.

        The effective values come from ``TorchIOQueueConfig.from_training_config``,
        the resolver the data pipeline director itself calls, so this check and
        the build cannot disagree about the ``samples_per_volume`` an arm that
        sets only the legacy ``sampling.*`` fields ends up with (non-negotiable
        17: one owner per resolved value).
        """
        check_name = "slice_level_records_queue_shape"
        data = getattr(config, "data", None)
        if data is None or getattr(data, "dataset_type", None) != "m4raw":
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message="n/a: not an m4raw arm (slice_level_records is m4raw-only).",
                severity="info",
                category="data",
            )
        if not data.slice_level_records:
            return HealthCheckResult(
                passed=True,
                check_name=check_name,
                message="n/a: data.slice_level_records is off (one record per group).",
                severity="info",
                category="data",
            )

        from spectramr.data.builders.torchio_queue_builder import TorchIOQueueConfig

        try:
            queue = TorchIOQueueConfig.from_training_config(data)
        except Exception as exc:  # a resolver that refuses the config is the finding
            return HealthCheckResult(
                passed=False,
                check_name=check_name,
                message=(
                    "data.slice_level_records is on but the queue configuration could "
                    f"not be resolved: {type(exc).__name__}: {exc}"
                ),
                severity="error",
                category="data",
                yaml_keys=["data.slice_level_records", "data.sampling"],
                fix_hint="Fix the sampling/loader block the error names.",
            )

        patch = tuple(queue.patch_size)
        depth = int(patch[2]) if len(patch) >= 3 else 1
        problems: list[str] = []
        yaml_keys = ["data.slice_level_records"]
        if depth != 1:
            problems.append(
                f"sampling.patch_size={list(patch)} has depth {depth}; a slice record is a "
                "depth-1 subject, so the queue's patch filter would drop every record"
            )
            yaml_keys.append("data.sampling.patch_size")
        if queue.sampler_type == "full":
            spv_note = "train sampler 'full' bypasses the queue (samples_per_volume unread)"
        elif queue.samples_per_volume != 1:
            problems.append(
                f"samples_per_volume={queue.samples_per_volume} draws that many patches of "
                "the SAME slice per epoch (identical copies under a full-slice patch, "
                "k-space crops under a smaller one); a slice record serves one sample"
            )
            yaml_keys.extend(
                ["data.modes.train.sampler.samples_per_volume", "data.sampling.samples_per_volume"]
            )
            spv_note = ""
        else:
            spv_note = "samples_per_volume=1"

        if problems:
            return HealthCheckResult(
                passed=False,
                check_name=check_name,
                message="data.slice_level_records=true but " + "; ".join(problems) + ".",
                severity="error",
                category="data",
                yaml_keys=yaml_keys,
                fix_hint=(
                    "Give sampling.patch_size a depth of 1 and set "
                    "data.modes.train.sampler.samples_per_volume: 1 (or sampler type "
                    "'full'); an epoch then visits every slice once. Or drop "
                    "data.slice_level_records to keep the per-group index."
                ),
            )
        return HealthCheckResult(
            passed=True,
            check_name=check_name,
            message=(
                f"slice-level records: patch depth 1, {spv_note}; every slice is served "
                "once per epoch."
            ),
            severity="info",
            category="data",
            yaml_keys=yaml_keys,
        )

    def check_transform_names_are_registered(self, config: TrainingSettings) -> HealthCheckResult:
        """Every ``data.processing.transforms`` name must resolve in the registry.

        Same shape as :meth:`check_metric_names_are_registered`, and for the
        same reason. ``data.processing.transforms`` was typed
        ``list[dict[str, Any]]`` with no validator, and its only consumer
        matched the single literal ``"graph_encoding"`` and stopped -- so every
        other declaration validated at load and was then silently discarded.
        Committed arms named for ``slice_profile``, ``synthetic_lesion`` and
        ``scout_acquisition`` trained without the mechanism they are named for,
        and the four arms spelling a dotted import path never had a resolver at
        all. Registry membership makes that a Tier-1 error instead of a
        silently-missing transform (pitfall #16 behind pitfall #15).

        Skips when the list is empty, i.e. for almost every arm.
        """
        declared = getattr(
            getattr(getattr(config, "data", None), "processing", None),
            "transforms",
            None,
        )
        if not declared:
            return HealthCheckResult(
                passed=True,
                check_name="transform_names_are_registered",
                message="data.processing.transforms not used.",
                severity="info",
            )
        try:
            from spectramr.data.transforms.registry import list_transforms
        except Exception as e:
            return HealthCheckResult(
                passed=True,
                check_name="transform_names_are_registered",
                message=f"transform registry not importable; skipping ({e}).",
                severity="info",
            )
        known = set(list_transforms())
        unknown: list[str] = []
        unnamed = 0
        for entry in declared:
            name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
            if not name:
                unnamed += 1
            elif name not in known:
                unknown.append(str(name))
        if not unknown and not unnamed:
            return HealthCheckResult(
                passed=True,
                check_name="transform_names_are_registered",
                message=f"all {len(declared)} declared transform(s) are registered.",
                severity="info",
            )
        parts = []
        if unknown:
            parts.append(f"{len(unknown)} unregistered name(s): {unknown}")
        if unnamed:
            parts.append(f"{unnamed} entry/entries with no 'name' key")
        return HealthCheckResult(
            passed=False,
            check_name="transform_names_are_registered",
            message=(
                f"data.processing.transforms has {'; '.join(parts)}. "
                f"Registered: {sorted(known)}. Such an entry used to be "
                "silently discarded, so the run reported success without the "
                "transform. Dotted import paths are not supported."
            ),
            severity="error",
        )

    def check_metric_names_are_registered(self, config: TrainingSettings) -> HealthCheckResult:
        """Every declared metric name must resolve in ``MetricsRegistry``.

        This is what makes the list safe to prefer over the 86 ``compute_*``
        flags. A flag could name a metric that does not exist and simply do
        nothing -- **21** of them do, and ``compute_advanced_metrics`` defaults
        to ``True`` while **734** arms set it explicitly and NOTHING reads it.
        (Both figures were stale here -- 17 and 249 -- until re-measured
        2026-08-02; CLAUDE.md carries the reproducer, pinned by
        ``tests/unit/core/metrics/test_flag_map.py``.) With a list, registry
        membership is the validator, so the same
        mistake is a startup error instead of a silently-missing measurement
        (pitfall #18: the arm grades on a metric the run never computes).

        Skips when both lists are empty, i.e. for every arm still on the flags.

        **Two surfaces, not one.** ``metrics.compute`` configures the TRAINING
        metrics computer; ``validation.scoring.compute`` configures the
        VALIDATION one (``mixins/metrics_mixin.py`` prefers it when non-empty),
        and they are routinely different lists on the same arm. This check read
        only the first until 2026-09-16, so a typo on the surface that grades an
        arm's headline numbers was not an error -- it was a column that never
        appeared. That is the same shape as #2116, one layer up: the name is
        accepted, nothing computes it, and the run reports success.
        """
        metrics_compute = getattr(getattr(config, "metrics", None), "compute", None) or []
        scoring = getattr(getattr(config, "validation", None), "scoring", None)
        scoring_compute = getattr(scoring, "compute", None) or []
        #: ``(surface, name)`` so the error names the key the reader has to edit.
        declared_by_surface = [
            *(("metrics.compute", n) for n in metrics_compute),
            *(("validation.scoring.compute", n) for n in scoring_compute),
        ]
        declared = [n for _, n in declared_by_surface]
        if not declared:
            return HealthCheckResult(
                passed=True,
                check_name="metric_names_are_registered",
                message=(
                    "metrics.compute / validation.scoring.compute not used; "
                    "arm is on the compute_* flags."
                ),
                severity="info",
            )
        try:
            from spectramr.core.metrics import MetricsRegistry
        except Exception as e:
            return HealthCheckResult(
                passed=True,
                check_name="metric_names_are_registered",
                message=f"MetricsRegistry not importable; skipping ({e}).",
                severity="info",
            )
        known = {k.lower() for k in getattr(MetricsRegistry, "_metrics", {})}
        known |= {k.lower() for k in getattr(MetricsRegistry, "_aliases", {})}
        unknown = [(s_, n) for s_, n in declared_by_surface if str(n).lower() not in known]
        if not unknown:
            surfaces = sorted({s_ for s_, _ in declared_by_surface})
            return HealthCheckResult(
                passed=True,
                check_name="metric_names_are_registered",
                message=(
                    f"all {len(declared)} declared metric(s) are registered "
                    f"({', '.join(surfaces)})."
                ),
                severity="info",
            )
        offending = sorted({s_ for s_, _ in unknown})
        return HealthCheckResult(
            passed=False,
            check_name="metric_names_are_registered",
            message=(
                f"{len(unknown)} unregistered metric(s): "
                f"{[f'{s_}: {n}' for s_, n in unknown]}. The run would report "
                "success while never computing them."
            ),
            severity="error",
            category="metrics_misconfiguration",
            yaml_keys=offending,
            fix_hint=(
                "Use a registered name (or a registered alias). "
                f"{len(known)} names are available; `MetricsRegistry.list_available()` "
                "enumerates them."
            ),
        )

    def check_repetition_count_is_achievable(self, config: TrainingSettings) -> HealthCheckResult:
        """``model_kwargs.num_repetitions`` must match what the data can supply.

        ``ComplexRepetitionFusion`` sizes a 1x1 complex conv at
        ``num_physical_coils * num_repetitions``, so the declared count is a hard
        shape contract. M4Raw ships **3** repetitions for T1/T2 and **2** for
        FLAIR, so ``4`` -- the value four arms declared -- is satisfiable by no
        contrast at all (#1173).

        Two properties of this check are deliberate:

        * **It reads the DECLARED value, and only fires when one was declared.**
          Applicability is gated on the same opt-in that now gates building the
          layer, which keeps the red count equal to the set of arms that
          actually asked for repetition fusion. The constructor's old
          ``kwargs.get("num_repetitions", 4)`` default is exactly the silent
          substitution non-negotiable 3 forbids, so there is no defaulted value
          left for this check to bless.
        * **The rep counts come from the producer**, ``m4raw_dataset``'s
          ``M4RAW_REPETITIONS_BY_CONTRAST``, not from a literal copied into this
          checker. One owner per invariant (non-negotiable 17); a second copy
          here would drift silently and each would pass its own tests.

        The sharpest outcome is the heterogeneous one: because the achievable
        counts differ ACROSS contrasts, an arm spanning T1/T2 and FLAIR has **no
        correct literal** -- 3 is wrong for FLAIR and 2 is wrong for T1/T2. Such
        an arm has to resolve the count from the manifest per sample rather than
        declare it, which is why the fix is an owner decision and not a
        find-and-replace.
        """
        model_cfg = getattr(config, "model", None)
        kwargs = getattr(model_cfg, "model_kwargs", None) or {}
        if not isinstance(kwargs, dict) or "num_repetitions" not in kwargs:
            return HealthCheckResult(
                passed=True,
                check_name="repetition_count_is_achievable",
                message="model_kwargs.num_repetitions not declared; nothing to validate.",
                severity="info",
            )

        declared = kwargs["num_repetitions"]
        data_cfg = getattr(config, "data", None)
        dataset_type = str(getattr(data_cfg, "dataset_type", "") or "")

        # Only the M4Raw path reads repetitions at all. UniversalMRIDataset --
        # which serves `dataset_type: kspace` -- has no repetition handling
        # whatsoever, so a declaration there is inert rather than wrong.
        if "m4raw" not in dataset_type.lower():
            return HealthCheckResult(
                passed=False,
                check_name="repetition_count_is_achievable",
                message=(
                    f"model_kwargs.num_repetitions={declared!r} is declared, but "
                    f"dataset_type={dataset_type!r} serves no repetitions: only the "
                    "M4Raw path reads them (UniversalMRIDataset, which backs "
                    "`kspace`/`fastmri_*`, has no repetition handling). The knob "
                    "sizes a fusion layer that this data can never feed."
                ),
                severity="error",
                category="model_data_mismatch",
                yaml_keys=["model.model_kwargs.num_repetitions", "data.dataset_type"],
                fix_hint=(
                    "Drop model_kwargs.num_repetitions (and data.use_repetitions, "
                    "which this dataset also ignores), or switch to "
                    "dataset_type: m4raw if repetitions are genuinely wanted."
                ),
            )

        try:
            from spectramr.data.datasets.m4raw_dataset import (
                M4RAW_REPETITIONS_BY_CONTRAST,
            )
        except Exception as e:  # pragma: no cover - import guard
            return HealthCheckResult(
                passed=True,
                check_name="repetition_count_is_achievable",
                message=f"M4Raw repetition map not importable; skipping ({e}).",
                severity="info",
            )

        # Which contrasts does this arm actually span?
        #
        # ``data.multi_contrast.contrast_map`` is the only field that names them,
        # and it MUST be read through ``model_fields_set`` rather than by value.
        # Its default is a fully-populated ``{T1, T2, FLAIR, PD}`` materialised on
        # every arm, so a truthiness test on it is a tautology -- every config
        # would look like it declared all four contrasts, and this check could
        # only ever reach one branch (the `model_dump` presence trap).
        #
        # ``data.contrasts`` does NOT exist and must not be reached for: that
        # field lives on ``DataPairingConfigSchema``, a different block, so
        # ``getattr(data_cfg, "contrasts", None)`` is a permanent ``None`` -- a
        # read that looks like contrast resolution and resolves nothing.
        names: list[str] = []
        mc = getattr(data_cfg, "multi_contrast", None)
        if mc is not None and "contrast_map" in getattr(mc, "model_fields_set", set()):
            names = [str(c).upper() for c in (getattr(mc, "contrast_map", None) or {})]

        if names:
            # A contrast absent from the producer map (e.g. PD, whose count M4Raw
            # documents as "variable") is UNKNOWN, not zero -- excluded from the
            # achievable set and reported, never defaulted.
            unknown = [n for n in names if n not in M4RAW_REPETITIONS_BY_CONTRAST]
            achievable = {
                M4RAW_REPETITIONS_BY_CONTRAST[n]
                for n in names
                if n in M4RAW_REPETITIONS_BY_CONTRAST
            }
        else:
            # The arm did not restrict its contrasts, so it draws on the whole
            # M4Raw corpus and must cope with every count that corpus contains.
            unknown = []
            achievable = set(M4RAW_REPETITIONS_BY_CONTRAST.values())
        declared_contrasts = names or None

        if not achievable:
            return HealthCheckResult(
                passed=True,
                check_name="repetition_count_is_achievable",
                message=(
                    f"no contrast with a known repetition count "
                    f"(declared: {declared_contrasts!r}); cannot validate "
                    f"num_repetitions={declared!r}."
                ),
                severity="info",
            )

        if len(achievable) > 1:
            detail = ", ".join(
                f"{c}={M4RAW_REPETITIONS_BY_CONTRAST[c]}"
                for c in sorted(M4RAW_REPETITIONS_BY_CONTRAST)
            )
            return HealthCheckResult(
                passed=False,
                check_name="repetition_count_is_achievable",
                message=(
                    f"model_kwargs.num_repetitions={declared!r} is a single literal, "
                    f"but this arm spans contrasts whose repetition counts DIFFER "
                    f"({detail}). No literal is correct for all of them: the fusion "
                    "layer would be mis-sized for whichever contrast it does not "
                    "match."
                ),
                severity="error",
                category="model_data_mismatch",
                yaml_keys=[
                    "model.model_kwargs.num_repetitions",
                    "data.multi_contrast.contrast_map",
                ],
                fix_hint=(
                    "Resolve the repetition count per sample from the manifest "
                    "instead of declaring it, or restrict the arm to contrasts "
                    "that share one count (T1+T2 = 3)."
                ),
            )

        only = next(iter(achievable))
        if declared == only:
            return HealthCheckResult(
                passed=True,
                check_name="repetition_count_is_achievable",
                message=f"num_repetitions={declared} matches the data ({only}).",
                severity="info",
            )

        note = f" (contrasts with unknown counts, excluded: {unknown})" if unknown else ""
        return HealthCheckResult(
            passed=False,
            check_name="repetition_count_is_achievable",
            message=(
                f"model_kwargs.num_repetitions={declared!r} but this arm's data "
                f"supplies {only}{note}. ComplexRepetitionFusion sizes a conv at "
                f"num_physical_coils * num_repetitions, so the mismatch is a shape "
                "contract the data cannot satisfy."
            ),
            severity="error",
            category="model_data_mismatch",
            yaml_keys=["model.model_kwargs.num_repetitions"],
            fix_hint=f"Set model_kwargs.num_repetitions: {only}, or drop the key.",
        )

    def check_loss_domain_block_match(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Block-level ``@register_loss(domain=)`` check.

        For each loss declared under a ``losses.*_losses`` list, resolve its
        registered contract through ``get_loss_capabilities`` -- the single
        retrieval surface -- and reject a placement whose domain cannot match.

        Surfaced by R20 (silent fallback): a kspace loss placed under
        image_losses compiles fine but produces meaningless gradients.

        Three outcomes, and the distinction between the last two is the point:

        * a concrete ``domain`` that differs from the block's  -> ERROR
        * ``domain_agnostic``  -> pass, permanently. ``l1``, ``l2``, ``bce`` and
          the other elementwise distances are defined on any tensor, and the
          corpus declares ``l1`` under ``image_losses`` 334 times AND under
          ``kspace_losses`` 7 times. Both are correct.
        * no annotation at all -> pass, provisionally. 104 of 214 registrations
          carry no domain, so this branch is where the check has no opinion
          rather than a favourable one.

        The agnostic branch is written out rather than left to fall through the
        unannotated one. It used to be incidental -- "agnostic" simply failed to
        map onto the ``Domain`` literal and arrived as ``None`` -- so tightening
        the unannotated branch would have silently started rejecting the
        generic losses.
        """
        results: list[HealthCheckResult] = []
        losses_cfg = getattr(config, "losses", None)
        if losses_cfg is None:
            return results
        try:
            from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS
            from spectramr.models.losses.registry import (
                compatible_domains,
                get_loss_capabilities,
            )
        except Exception as e:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="loss_domain_block_match",
                    message=f"LossRegistry not importable; skipping ({e}).",
                    severity="info",
                )
            )
            return results
        any_mismatch = False
        for block_attr, expected in LOSS_LIST_DOMAINS.items():
            block = getattr(losses_cfg, block_attr, None) or []
            for entry in block:
                name = getattr(entry, "name", None) or (
                    entry.get("name") if isinstance(entry, dict) else None
                )
                if not name:
                    continue
                caps = get_loss_capabilities(name)
                if caps is None:
                    # Unregistered, or registered with no metadata. Reachability
                    # is check_loss_names_are_registered's job, not this one.
                    continue
                if caps.domain_agnostic:
                    continue
                if caps.domain is None:
                    continue
                if caps.domain == expected.value:
                    continue
                if expected.value in compatible_domains(name):
                    continue
                any_mismatch = True
                results.append(
                    HealthCheckResult(
                        passed=False,
                        check_name="loss_domain_block_match",
                        message=(
                            f"Loss '{name}' is registered with domain="
                            f"{caps.domain!r} but is declared under "
                            f"losses.{block_attr} (which grades in "
                            f"{expected.value!r}). Either move it to the "
                            "matching block or update its decorator's "
                            "compatible_with list."
                        ),
                        severity="error",
                        category="loss_misconfiguration",
                        yaml_keys=[f"losses.{block_attr}"],
                        fix_hint=(
                            f"Move '{name}' to the list grading in "
                            f"{caps.domain!r}, or pass "
                            f"compatible_with=[{expected.value!r}] in its "
                            "@register_loss decorator."
                        ),
                    )
                )
        if not any_mismatch:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="loss_domain_block_match",
                    message="All loss-block placements match registered domains.",
                    severity="info",
                )
            )
        return results

    # ------------------------------------------------------------------

    def check_workflow_declared(self, config: TrainingSettings) -> HealthCheckResult:
        """Require a coherent ``workflow:`` (imaging-regime × task) block.

        The ``workflow`` field is optional on :class:`TrainingSettings`
        (Pydantic never breaks), but a declared regime is what lets the
        rest of the audit ladder reason about axes, forward operators, and
        maturity. This check is the "optional now, required later" seam:

        - **absent** ``workflow`` (or missing ``regime``) → error;
        - regime whose :class:`Maturity` is ``STUB`` → error (the framework
          cannot run it);
        - a ``task`` the regime does not support → error.
        """
        from spectramr.config.schemas.enums import Maturity
        from spectramr.domain.workflows import WORKFLOW_PROFILES

        name = declared_regime(config)
        if name is None:
            # ADVISORY, not an error — deliberately. This check was authored on the
            # `public`/main branch, where `experiments/` is stripped from the tree, so
            # "absent => error" cost nothing there. On dev it is the difference between
            # a clean audit and 1,465 red arms: not one experiment YAML predates this
            # feature with a workflow: block, and this port adds none. Erroring here
            # would not "enforce" the contract, it would simply make `spectramr audit`
            # useless on every existing arm overnight.
            #
            # The guards that carry the real signal stay hard errors below (a STUB
            # regime, an unsupported regime x task pair) — those only fire on an arm
            # that DID declare, and a wrong declaration must never pass.
            #
            # Ratchet to severity="error" once the cohorts are annotated. See #283.
            return HealthCheckResult(
                passed=True,
                check_name="workflow_declared",
                message=(
                    "No workflow: block declared — the imaging regime x task contract "
                    "is un-annotated for this arm, so the axis/forward-operator/maturity "
                    "checks cannot run on it. Advisory until the cohorts are annotated "
                    "(#283), then this becomes an error."
                ),
                severity="info",
                category="workflow",
                yaml_keys=["workflow.regime"],
                # `regime:`, not `name:` -- the field was renamed on 2026-07-31
                # with posture="raise", so the old spelling is a hard
                # ValidationError, not a fold. This hint fires on every arm
                # lacking a workflow: block, i.e. during the exact migration it
                # is meant to help with.
                fix_hint=(
                    "Add e.g.\n  workflow:\n    regime: mri_structural\n    task: reconstruction"
                ),
            )

        profile = WORKFLOW_PROFILES.get(name)
        if profile is None:  # pragma: no cover - enum ⇒ profile invariant
            return HealthCheckResult(
                passed=False,
                check_name="workflow_declared",
                message=f"Regime {name.value!r} has no registered WorkflowProfile.",
                severity="error",
                category="workflow",
                yaml_keys=["workflow.regime"],
            )

        if profile.maturity is Maturity.STUB:
            return HealthCheckResult(
                passed=False,
                check_name="workflow_declared",
                message=(
                    f"Regime {name.value!r} is a STUB — the framework has no "
                    "forward operator, losses or strategy for it. Training and "
                    "inference will raise WorkflowNotImplementedError."
                ),
                severity="error",
                category="workflow",
                yaml_keys=["workflow.regime"],
                fix_hint="Pick a LIVE/PARTIAL regime, or implement the stub first.",
            )

        task = declared_task(config)
        if task is not None and task not in profile.supported_tasks:
            supported = sorted(t.value for t in profile.supported_tasks)
            return HealthCheckResult(
                passed=False,
                check_name="workflow_declared",
                message=(
                    f"Task {task.value!r} is not supported by regime "
                    f"{name.value!r}. Supported: {supported}."
                ),
                severity="error",
                category="workflow",
                yaml_keys=["workflow.task"],
                fix_hint=f"Use one of {supported}.",
            )

        return HealthCheckResult(
            passed=True,
            check_name="workflow_declared",
            message=(
                f"workflow={name.value!r} (maturity={profile.maturity.value})"
                + (f", task={task.value!r}" if task is not None else "")
            ),
            severity="info",
        )

    def check_workflow_required_axes(self, config: TrainingSettings) -> HealthCheckResult:
        """Reject a regime whose required axes the data cannot expose.

        The machine-readable form of pitfall #19 ("hypothesis untestable on
        this data"): ``mri_functional`` needs a ``TEMPORAL`` axis, but
        ``dataset_type: image`` exposes none — that arm can never test its
        hypothesis. The check fires only when the ``dataset_type``'s axis
        exposure is *known*; an arm that neither declares nor is annotated is
        skipped, never guessed.

        Axes are resolved DECLARED-first: ``data.bart.bart_dim_map`` is a
        per-arm, schema-validated statement of what the acquisition carries, and
        it outranks the per-type annotation in
        ``spectramr.data.datasets.axis_exposure``. That ordering is what made the
        rule reach ``bart_kspace`` at all -- its axes differ between arms, so it
        has no table row and every one of its arms used to skip.
        """
        name = declared_regime(config)
        if name is None:
            return HealthCheckResult(
                passed=True,
                check_name="workflow_required_axes",
                message="No workflow declared; required-axes check skipped.",
                severity="info",
            )

        from spectramr.data.datasets.axis_exposure import (
            declared_axes_for,
            resolve_axes_for,
        )
        from spectramr.domain.workflows import WORKFLOW_PROFILES

        profile = WORKFLOW_PROFILES.get(name)
        required = profile.required_axes if profile else frozenset()
        if not required:
            return HealthCheckResult(
                passed=True,
                check_name="workflow_required_axes",
                message=f"Regime {name.value!r} requires no non-spatial axes.",
                severity="info",
            )

        data_cfg = getattr(config, "data", None)
        dataset_type = getattr(data_cfg, "dataset_type", None)
        # A per-arm DECLARATION outranks the per-type annotation. ``bart_dim_map``
        # is validated by BartConfigSchema and describes THIS arm's acquisition;
        # DATASET_TYPE_AXES generalises over a whole corpus. ``bart_kspace`` has no
        # row in that table and cannot be given one — its arms disagree about which
        # axes they carry — so before this composition every bart arm SKIPPED,
        # which is why the rule never saw the one fact it exists to consume.
        # One resolver, shared with the batch contract (``TrainingBatch.axes``).
        # The composition used to be inlined right here; a second hand-written
        # copy is how the audit and the tensor start disagreeing about the same
        # arm. ``declared_axes_for`` is still called separately, but only to
        # NAME the route in the message -- it does not decide the answer.
        exposed = resolve_axes_for(data_cfg)
        declared = declared_axes_for(data_cfg)
        source = (
            "declared in data.bart.bart_dim_map"
            if declared is not None
            else f"annotated for dataset_type={dataset_type!r}"
        )
        if exposed is None:
            # Unannotated dataset_type and no per-arm declaration — cannot vouch
            # for its axes, so skip.
            return HealthCheckResult(
                passed=True,
                check_name="workflow_required_axes",
                message=(
                    f"dataset_type={dataset_type!r} has no declared axis "
                    "exposure; required-axes check skipped."
                ),
                severity="info",
            )

        # ANY of the required axes satisfies the regime, which is what this
        # check's own error message has always said ("expose none of them").
        # It computed ``required - exposed`` -- an all-of test -- and the two
        # agreed only because every profile declared 0 or 1 axis. That stopped
        # being true when ``mri_quantitative`` came to mean "echo OR flip_angle"
        # (#1020). See WorkflowProfile.required_axes.
        if not (required & exposed):
            missing_names = sorted(a.value for a in required)
            return HealthCheckResult(
                passed=False,
                check_name="workflow_required_axes",
                message=(
                    f"Regime {name.value!r} requires axes {missing_names} but "
                    f"the axes {source} expose none of them — the arm's "
                    "hypothesis is untestable on this data (pitfall #19)."
                ),
                severity="error",
                category="workflow",
                yaml_keys=["workflow.regime", "data.dataset_type"],
                fix_hint=(
                    f"Use a dataset_type that exposes {missing_names} (e.g. a "
                    "temporal/multi-echo/spectral dataset), declare the axis in "
                    "data.bart.bart_dim_map if this is a BART acquisition that "
                    "really carries it, or pick a regime whose required axes "
                    "this data provides."
                ),
            )

        return HealthCheckResult(
            passed=True,
            check_name="workflow_required_axes",
            message=(
                # Name what SATISFIED the requirement, not the whole required
                # set: under any-of semantics those differ, and printing the set
                # would tell a B1+ arm it declares an echo axis it does not have.
                f"The axes {source} include "
                f"{sorted(a.value for a in (required & exposed))}, which "
                f"satisfies {name.value!r} (needs any of "
                f"{sorted(a.value for a in required)})."
            ),
            severity="info",
        )

    def check_workflow_spatial_rank(self, config: TrainingSettings) -> HealthCheckResult:
        """Reconcile three statements of spatial rank, in order of cost.

        ``mri_functional`` is a rank-3 regime (3-D volume over time); declaring
        it on ``dataset_type: 2d`` is incoherent.

        There are three sources, not two, since ``workflow.spatial_rank`` became
        declarable: the regime's ``spatial_ranks``, the rank the ``dataset_type``
        provides, and the rank the author *claims*. They are checked in that
        order, and the order matters — the declared-vs-profile comparison needs
        neither the dataset nor its annotation, so it must happen **before** the
        "unannotated ``dataset_type`` → skip" branch. Putting it after would let
        an unannotated dataset swallow an outright contradiction (``regime:
        mri_functional`` + ``spatial_rank: 2``), which is precisely the shape of
        error a declared field exists to catch.

        Declared-vs-dataset is the most interesting of the three: an arm
        claiming rank 3 while feeding 2-D slices is a claim the run will quietly
        fail to honour.
        """
        name = declared_regime(config)
        if name is None:
            return HealthCheckResult(
                passed=True,
                check_name="workflow_spatial_rank",
                message="No workflow declared; spatial-rank check skipped.",
                severity="info",
            )

        from spectramr.data.datasets.axis_exposure import spatial_rank_for
        from spectramr.domain.workflows import WORKFLOW_PROFILES

        profile = WORKFLOW_PROFILES.get(name)
        ranks = profile.spatial_ranks if profile else frozenset()
        declared = declared_spatial_rank(config)

        if declared is not None and ranks and declared not in ranks:
            return HealthCheckResult(
                passed=False,
                check_name="workflow_spatial_rank",
                message=(
                    f"workflow.spatial_rank={declared} is not one of "
                    f"{name.value!r}'s spatial ranks {sorted(ranks)}."
                ),
                severity="error",
                category="workflow",
                yaml_keys=["workflow.regime", "workflow.spatial_rank"],
                fix_hint=(
                    f"Declare a rank in {sorted(ranks)}, pick a regime that "
                    "accepts this rank, or omit spatial_rank and let the "
                    "dataset determine it."
                ),
            )

        if not ranks:
            return HealthCheckResult(
                passed=True,
                check_name="workflow_spatial_rank",
                message=f"Regime {name.value!r} declares no spatial-rank constraint.",
                severity="info",
            )

        data_cfg = getattr(config, "data", None)
        dataset_type = getattr(data_cfg, "dataset_type", None)
        rank = spatial_rank_for(dataset_type)
        if rank is None:
            return HealthCheckResult(
                passed=True,
                check_name="workflow_spatial_rank",
                message=(
                    f"dataset_type={dataset_type!r} has no declared spatial "
                    "rank; spatial-rank check skipped."
                    + (
                        f" workflow.spatial_rank={declared} satisfies "
                        f"{name.value!r} ranks {sorted(ranks)}."
                        if declared is not None
                        else ""
                    )
                ),
                severity="info",
            )

        if declared is not None and declared != rank:
            return HealthCheckResult(
                passed=False,
                check_name="workflow_spatial_rank",
                message=(
                    f"Arm declares workflow.spatial_rank={declared}, but "
                    f"dataset_type={dataset_type!r} provides rank {rank}. The "
                    "data cannot honour the claim."
                ),
                severity="error",
                category="workflow",
                yaml_keys=["workflow.spatial_rank", "data.dataset_type"],
                fix_hint=(
                    f"Use a dataset_type of rank {declared}, or declare "
                    f"spatial_rank: {rank} to match the data."
                ),
            )

        if rank not in ranks:
            return HealthCheckResult(
                passed=False,
                check_name="workflow_spatial_rank",
                message=(
                    f"Regime {name.value!r} requires spatial rank in "
                    f"{sorted(ranks)} but dataset_type={dataset_type!r} provides "
                    f"rank {rank}."
                ),
                severity="error",
                category="workflow",
                yaml_keys=["workflow.regime", "data.dataset_type"],
                fix_hint=(
                    f"Use a dataset_type of rank {sorted(ranks)}, or pick a "
                    "regime that accepts this data's rank."
                ),
            )

        return HealthCheckResult(
            passed=True,
            check_name="workflow_spatial_rank",
            message=(
                f"dataset_type={dataset_type!r} (rank {rank}) satisfies "
                f"{name.value!r} ranks {sorted(ranks)}."
            ),
            severity="info",
        )

    def check_workflow_signal_domain(self, config: TrainingSettings) -> HealthCheckResult:
        """Reject a model that consumes a signal the regime does not produce.

        ``WorkflowProfile.signal_domains`` declares what the regime's signal
        **is** — ``mri_spectroscopy`` yields a ``spectrum`` (an FID), the MR
        image/k-space regimes yield ``image``/``kspace``/``complex_image``. A
        model declares what it **consumes** in ``ModelCapabilities.input_domain``.
        Those are the same question asked from two ends, so a disjoint pair is a
        real mismatch: an image-reconstruction UNet pointed at a free induction
        decay, which would train happily and mean nothing.

        This was the last field on ``WorkflowProfile`` that was declared but
        never asserted — the same shape as ``forward_operator`` before the ledger
        walked it, and as PARTIAL before its branch existed. It was skipped in
        the original pass for false-positive risk; the risk was real and is what
        determines the direction checked here.

        **``input_domain``, deliberately not ``output_domain``.** ``signal_domains``
        is about the acquisition, not the prediction. The parameter-mapping arms
        all *emit* something other than their regime's signal —
        ``MRSQuantificationStrategy`` consumes a ``spectrum`` and emits
        ``[B, 4M, H, W]`` resonance maps (domain ``image``); perfusion consumes a
        DCE series and emits ``(Ktrans, ve, vp)``. Checking the output would
        reject the very arms these regimes exist for.

        Skips when either side is unannotated, never guesses — the escape hatch
        ``check_data_model_compatibility`` already establishes. Fires only on a
        *disjoint* pair, so a model that handles several domains passes if any
        one of them fits.
        """
        name = declared_regime(config)
        if name is None:
            return HealthCheckResult(
                passed=True,
                check_name="workflow_signal_domain",
                message="No workflow declared; signal-domain check skipped.",
                severity="info",
            )

        from spectramr.domain.workflows import WORKFLOW_PROFILES
        from spectramr.infrastructure.validation.spec_card import _derive_model_form

        profile = WORKFLOW_PROFILES.get(name)
        declared = profile.signal_domains if profile else frozenset()
        if not declared:
            return HealthCheckResult(
                passed=True,
                check_name="workflow_signal_domain",
                message=f"Regime {name.value!r} declares no signal domains.",
                severity="info",
            )

        # `workflow.signal_domain` NARROWS the regime's set to the one this arm
        # says it consumes. Checked against the profile first: a domain the
        # regime does not produce is wrong regardless of what the model is, and
        # this comparison needs no model annotation, so it must not sit behind
        # the "model declares no input_domain -> skip" branch below.
        arm_domain = declared_signal_domain(config)
        if arm_domain is not None:
            if arm_domain.value not in declared:
                return HealthCheckResult(
                    passed=False,
                    check_name="workflow_signal_domain",
                    message=(
                        f"workflow.signal_domain={arm_domain.value!r} is not "
                        f"produced by {name.value!r}, which yields "
                        f"{sorted(declared)}."
                    ),
                    severity="error",
                    category="workflow",
                    yaml_keys=["workflow.regime", "workflow.signal_domain"],
                    fix_hint=(
                        f"Declare one of {sorted(declared)}, or pick the regime "
                        "whose signal this arm actually consumes."
                    ),
                )
            # Narrowed: the model is now held to the ONE domain the arm claims,
            # not to the regime's whole set. `signal_domain` means *consumes*
            # (== ModelCapabilities.input_domain), never *emits* — reading it as
            # emits would reject every parameter-mapping arm, which is the
            # false positive this check's `input_domain` choice exists to avoid.
            declared = frozenset({arm_domain.value})

        model_form = _derive_model_form(config)
        caps = model_form.get("capabilities") if model_form.get("present") else None
        consumed = getattr(caps, "input_domain", None) if caps else None
        if consumed is None:
            return HealthCheckResult(
                passed=True,
                check_name="workflow_signal_domain",
                message=(
                    f"model_type={model_form.get('model_type')!r} declares no "
                    "input_domain; signal-domain check skipped."
                ),
                severity="info",
            )

        consumed_set = set(consumed) if isinstance(consumed, tuple) else {consumed}
        if consumed_set & set(declared):
            return HealthCheckResult(
                passed=True,
                check_name="workflow_signal_domain",
                message=(
                    f"model input_domain {sorted(consumed_set)} matches "
                    f"{name.value!r} signal_domains {sorted(declared)}."
                ),
                severity="info",
            )

        return HealthCheckResult(
            passed=False,
            check_name="workflow_signal_domain",
            message=(
                f"Regime {name.value!r} produces {sorted(declared)}, but "
                f"model_type={model_form.get('model_type')!r} consumes "
                f"{sorted(consumed_set)} — the two are disjoint, so this model "
                "cannot be reading this regime's signal."
            ),
            severity="error",
            category="workflow",
            yaml_keys=["workflow.regime", "model.model_type"],
            fix_hint=(
                f"Pick a model whose input_domain includes one of "
                f"{sorted(declared)}, or a regime whose signal this model "
                "consumes. If the model genuinely handles both, add the domain "
                "to its ModelCapabilities.input_domain tuple."
            ),
        )

    def check_workflow_dataset_signal_domain(self, config: TrainingSettings) -> HealthCheckResult:
        """Reject a model that consumes a domain THIS dataset cannot materialise.

        The dataset-level companion to :meth:`check_workflow_signal_domain`. That
        check compares the model's ``input_domain`` against the *regime's*
        ``signal_domains`` — a permissive set of everything the regime *can* be
        (``mri_structural`` spans ``{image, kspace, complex_image}``). A k-space model
        pointed at magnitude-only mrixfields data passes it, because ``kspace`` is a
        legal structural domain in the abstract — yet the mrixfields loader never
        produces k-space, so that arm is dead on arrival.

        This check closes that gap: it compares the model's ``input_domain`` against
        the domain the *dataset_type* actually materialises
        (``data.datasets.axis_exposure.resolve_signal_domains_for``). A disjoint pair
        is a real mismatch the regime-level check cannot see.

        Resolved per ARM, not per ``dataset_type``: ``coil_processing_mode``
        (``rss_image`` / ``magnitude``) applies the IFFT inside the dataset's own
        transform pipeline, so those arms serve images however ``dataset_type``
        reads.

        Skips (never guesses) when: no workflow, the model declares no
        ``input_domain``, the ``dataset_type`` is unannotated, **or the arm declares
        adapters** — a ``pre_model`` adapter (e.g. ``fft_image_to_kspace``) can
        legitimately bridge an image dataset to a k-space model, and reasoning about
        that bridge is the adapter-chain composer's job, not this check's.
        """
        name = declared_regime(config)
        if name is None:
            return HealthCheckResult(
                passed=True,
                check_name="workflow_dataset_signal_domain",
                message="No workflow declared; dataset-signal-domain check skipped.",
                severity="info",
            )

        # A pre_model adapter may bridge the domain (e.g. fft_image_to_kspace), and the
        # adapter-chain composer owns that. Only pre_model, and only if it has at least
        # one ENABLED step: `adapters is not None` skipped on any adapters object at all,
        # so `adapters: {pre_metric: [...]}`, an all-disabled chain, or even `adapters: {}`
        # silently disarmed the check. That is a rule whose escape hatch is wider than
        # its rationale — the four non-pre_model hooks run after the model has already
        # consumed the data, so they cannot bridge its input domain.
        if _has_enabled_pre_model_adapter(getattr(config, "adapters", None)):
            return HealthCheckResult(
                passed=True,
                check_name="workflow_dataset_signal_domain",
                message=(
                    "Arm declares an enabled pre_model adapter that may bridge the data "
                    "domain; dataset-signal-domain check skipped (see adapter-chain "
                    "composer)."
                ),
                severity="info",
            )

        from spectramr.data.datasets.axis_exposure import resolve_signal_domains_for
        from spectramr.infrastructure.validation.spec_card import _derive_model_form

        data_cfg = getattr(config, "data", None)
        dataset_type = getattr(data_cfg, "dataset_type", None)
        # Arm-aware, not type-keyed: ``coil_processing_mode: rss_image`` moves the
        # IFFT inside the dataset's transform pipeline, so a `kspace` arm in that
        # mode serves IMAGES. Reading the type row alone reported 16 arms across 10
        # model families as unreadable when every one was correctly configured
        # (#1010).
        produced = resolve_signal_domains_for(data_cfg)
        from spectramr.data.datasets.axis_exposure import (
            IMAGE_DOMAIN_COIL_MODES,
            _coil_processing_mode,
        )

        _mode = _coil_processing_mode(data_cfg)
        _coil_mode = _mode if _mode in IMAGE_DOMAIN_COIL_MODES else ""
        if produced is None:
            return HealthCheckResult(
                passed=True,
                check_name="workflow_dataset_signal_domain",
                message=(
                    f"dataset_type={dataset_type!r} has no declared signal domain; "
                    "dataset-signal-domain check skipped."
                ),
                severity="info",
            )

        model_form = _derive_model_form(config)
        caps = model_form.get("capabilities") if model_form.get("present") else None
        consumed = getattr(caps, "input_domain", None) if caps else None
        if consumed is None:
            return HealthCheckResult(
                passed=True,
                check_name="workflow_dataset_signal_domain",
                message=(
                    f"model_type={model_form.get('model_type')!r} declares no "
                    "input_domain; dataset-signal-domain check skipped."
                ),
                severity="info",
            )

        consumed_set = set(consumed) if isinstance(consumed, tuple) else {consumed}
        if consumed_set & set(produced):
            return HealthCheckResult(
                passed=True,
                check_name="workflow_dataset_signal_domain",
                message=(
                    f"model input_domain {sorted(consumed_set)} matches "
                    f"dataset_type={dataset_type!r} signal domain {sorted(produced)}"
                    + (
                        f" (via coil_processing_mode={_coil_mode!r}, which moves the "
                        "IFFT into the dataset's transform pipeline)."
                        if _coil_mode
                        else "."
                    )
                ),
                severity="info",
            )

        return HealthCheckResult(
            passed=False,
            check_name="workflow_dataset_signal_domain",
            message=(
                f"dataset_type={dataset_type!r} materialises {sorted(produced)}, but "
                f"model_type={model_form.get('model_type')!r} consumes "
                f"{sorted(consumed_set)} — the two are disjoint, so this model cannot "
                "read what this dataset produces (the regime permits the model's "
                "domain in the abstract, but this data never yields it)."
            ),
            severity="error",
            category="workflow",
            yaml_keys=["workflow.regime", "data.dataset_type", "model.model_type"],
            fix_hint=(
                f"Point at data that materialises one of {sorted(consumed_set)}, pick a "
                f"model whose input_domain includes one of {sorted(produced)}, or add a "
                "pre_model adapter that bridges the two (e.g. fft_image_to_kspace)."
            ),
        )

    def check_workflow_component_regime(self, config: TrainingSettings) -> HealthCheckResult:
        """Reject a loss/metric whose regime tag excludes the arm's declared regime.

        The per-arm companion to the maturity ledger. The ledger asks a *corpus*
        question — "is regime R backed by at least one tagged component?" — walking
        every registered component. This asks the *arm* question: for an arm that
        declares ``workflow.regime = R``, does every loss and metric it actually USES
        fit R? A ``FLOW``-tagged divergence metric or a ``PERFUSION``-tagged
        tracer-kinetic loss on an ``mri_structural`` arm is a real mis-pairing that no
        other check catches — the tags exist, the ledger reads them corpus-wide, but
        nothing had ever matched them against the arm carrying the component.

        This is what makes the multi-valued ``workflows`` tag do per-arm work: a
        component tagged ``{FUNCTIONAL, DYNAMIC}`` passes on either regime, a
        single-tagged one only on its own, and an **untagged (agnostic)** component —
        the common case, e.g. ``l1`` / ``ssim`` / ``psnr`` — always skips, never
        guesses. Absent tag = skip, mirroring every other workflow check.

        Losses and metrics only: model tags are agnostic (``None``) across the whole
        registry today, and a strategy is chosen by class, not carried in a list —
        both are out of scope here and noted as extensions.
        """
        name = declared_regime(config)
        if name is None:
            return HealthCheckResult(
                passed=True,
                check_name="workflow_component_regime",
                message="No workflow declared; component-regime check skipped.",
                severity="info",
            )

        from spectramr.core.metrics.registry import MetricsRegistry
        from spectramr.models.losses.registry import LossRegistry

        def _tag(table: dict, component: str) -> frozenset | None:
            # Registries key on lower-cased canonical names; a miss (unknown or
            # untagged) yields None -> skip, the safe polarity. Alias-only names may
            # skip; that never false-positives.
            meta = table.get(component) or table.get(component.lower())
            return (meta or {}).get("workflows")

        declared: list[tuple[str, str]] = []  # (kind, name)
        losses_cfg = getattr(config, "losses", None)
        for attr in LOSS_LIST_DOMAINS:
            for entry in getattr(losses_cfg, attr, None) or []:
                if getattr(entry, "enabled", True) is False:
                    continue
                lname = (
                    getattr(entry, "name", None)
                    or getattr(entry, "loss_type", None)
                    or getattr(entry, "type", None)
                )
                if lname:
                    declared.append(("loss", str(lname)))
        val_cfg = getattr(config, "validation", None)
        for mname in (val_cfg.scoring.compute if val_cfg else None) or []:
            if mname:
                declared.append(("metric", str(mname)))

        mismatches: list[str] = []
        checked = 0
        for kind, component in declared:
            table = LossRegistry._loss_domains if kind == "loss" else MetricsRegistry._workflow_tags
            tags = _tag(table, component)
            if not tags:
                continue  # untagged / agnostic / unknown -> skip
            checked += 1
            if name not in tags:
                mismatches.append(f"{kind} {component!r} is tagged {sorted(r.value for r in tags)}")

        if mismatches:
            return HealthCheckResult(
                passed=False,
                check_name="workflow_component_regime",
                message=(
                    f"Arm declares workflow.regime={name.value!r}, but these components "
                    f"are tagged for other regimes: {'; '.join(mismatches)}. A "
                    "regime-specific loss/metric on a foreign regime is a mis-pairing "
                    "(its physics/units do not apply to this signal)."
                ),
                severity="error",
                category="workflow",
                yaml_keys=["workflow.regime", "losses", "validation.metrics"],
                fix_hint=(
                    "Remove the foreign-regime component, pick one tagged for "
                    f"{name.value!r} (or untagged/agnostic), or correct workflow.regime "
                    "if the regime itself is wrong. If the component genuinely serves "
                    "several regimes, add this regime to its workflows= tag."
                ),
            )

        return HealthCheckResult(
            passed=True,
            check_name="workflow_component_regime",
            message=(
                f"{checked} regime-tagged component(s) fit workflow.regime={name.value!r}"
                if checked
                else "No regime-tagged losses/metrics declared; nothing to match."
            ),
            severity="info",
        )

    def check_knob_applicability(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Reject a config block enabled outside the regime it is meaningful for.

        Catches the inert-knob trap: an ``mrf:`` block under ``mri_structural``,
        or ``data.quantitative`` enabled when the arm is not mapping
        parameters. Driven by the SSOT table in
        ``spectramr.domain.workflows.knobs``.
        """
        results: list[HealthCheckResult] = []
        regime = declared_regime(config)
        if regime is None:
            return results

        from spectramr.domain.workflows.knobs import KNOB_APPLICABILITY

        def _resolve(path: str) -> object | None:
            obj: object | None = config
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    return None
            return obj

        for knob in KNOB_APPLICABILITY:
            block = _resolve(knob.path)
            if block is None:
                continue
            enabled = (
                True if knob.enabled_key is None else bool(getattr(block, knob.enabled_key, False))
            )
            if not enabled:
                continue
            if regime not in knob.regimes:
                allowed = sorted(r.value for r in knob.regimes)
                results.append(
                    HealthCheckResult(
                        passed=False,
                        check_name="knob_applicability",
                        message=(
                            f"Config block {knob.path!r} is enabled but is only "
                            f"meaningful for regimes {allowed}; the declared "
                            f"regime is {regime.value!r}. It would be an inert "
                            "knob that silently does nothing."
                        ),
                        severity="error",
                        category="workflow",
                        yaml_keys=[knob.path],
                        fix_hint=(f"Remove {knob.path!r}, or declare a regime in {allowed}."),
                    )
                )

        if not results:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="knob_applicability",
                    message="All enabled regime-gated knobs match the declared regime.",
                    severity="info",
                )
            )
        return results

    # ──────────────────────────────────────────────────────────────────
    # Phase: complex-data discipline
    # ──────────────────────────────────────────────────────────────────

    def check_svd_compression_phase_safety(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        r"""Reject SVD coil compression configurations that lose phase.

        SVD-based coil compression operates on **complex** k-space:

        .. math::

            \tilde{y}_v = \sum_c A_{vc}\, y_c

        where the mixing matrix :math:`A` is complex-valued and the
        virtual coil :math:`\tilde{y}_v` retains both magnitude and
        phase. Two concrete code paths in this repo can silently
        invalidate that contract:

        1. ``coil_processing_mode: svd`` paired with a dataset that
           pre-stacks real/imag as channels (``2C``-channel layout)
           BEFORE the transform. The SVDCoilCompressionTransform then
           silently SKIPS the SVD (see
           ``src/data/transforms/coil_compression.py:200``), which
           leaves the model with un-compressed real-stacked channels —
           a CLAUDE.md #9 silent-fallback violation.

        2. Any pipeline that applies SVD per-component to real/imag
           streams independently. This breaks the covariance under
           unitary coil-space rotation that the SENSE-adjoint loss
           and the dual-domain complex-spatial-gradient loss depend
           on.

        We flag (1) at config-load time. The rule is: if SVD is
        requested AND ``model.in_channels`` is even with
        ``num_virtual_coils * 2 == in_channels`` AND no upstream
        complex→real adapter is declared, the data layout strongly
        suggests real-stacked input — likely SVD-skip silently.

        References
        ----------
        * Buehrer M., et al. "Array compression for MRI with large
          coil arrays." MRM 57:1131-1139, 2007.
        * Pruessmann K. P., et al. "SENSE: sensitivity encoding for
          fast MRI." MRM 42:952-962, 1999. (covariance under unitary
          coil-space rotation)
        """
        check_name = "svd_compression_phase_safety"
        data = getattr(config, "data", None)
        if data is None:
            return HealthCheckResult(True, check_name, "no data block to check", "info")
        coil_mode = (getattr(getattr(data, "coils", None), "processing_mode", "") or "").lower()
        if coil_mode != "svd":
            return HealthCheckResult(True, check_name, "non-SVD coil mode", "info")

        nv = int(getattr(getattr(data, "coils", None), "num_virtual_coils", 0) or 0)
        model = getattr(config, "model", None)
        in_ch = int(getattr(model, "in_channels", 0) or 0) if model is not None else 0

        # Detect declared upstream adapters that legitimately bridge
        # complex<->real-stacked formats (covers the explicit case).
        adapters = []
        for section in ("transforms", "adapters"):
            section_obj = getattr(data, section, None)
            if section_obj is not None:
                adapters.extend(getattr(section_obj, "pre_model", []) or [])
        adapter_names = {
            (a.get("name") if isinstance(a, dict) else getattr(a, "name", "")) for a in adapters
        }
        has_explicit_bridge = any(
            "complex_to_real" in (n or "").lower()
            or "real_imag_interleave" in (n or "").lower()
            or "complex_split" in (n or "").lower()
            for n in adapter_names
        )

        # Heuristic: 2 * num_virtual_coils == model.in_channels strongly
        # implies the dataset already real-stacked → SVDTransform will skip.
        if nv > 0 and in_ch == 2 * nv and not has_explicit_bridge:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"coil_processing_mode='svd' with num_virtual_coils={nv} and "
                    f"model.in_channels={in_ch} (=2*nv) suggests the dataset "
                    f"already real-stacks complex k-space as channels. The "
                    f"SVDCoilCompressionTransform will silently SKIP compression "
                    f"on real-valued input (CLAUDE.md #9 silent fallback) and "
                    f"phase information is lost for downstream SENSE-adjoint / "
                    f"dual-domain losses. Either (a) declare an explicit "
                    f"complex_to_real_imag_interleave adapter AFTER the SVD "
                    f"transform, or (b) drop coil_processing_mode='svd' if "
                    f"the data is already in real-stacked form."
                ),
                "error",
            )
        return HealthCheckResult(
            True,
            check_name,
            "SVD compression contract appears phase-safe",
            "info",
        )

    # ──────────────────────────────────────────────────────────────────
    # Phase: validation-image domain consistency (experiment_11 fix)
    # ──────────────────────────────────────────────────────────────────

    def check_validation_image_domain_safe(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        r"""Pre-flight guard for the kspace-cold-diffusion doubled-brain bug.

        When a model declares ``input_type: kspace`` but the dataset
        delivers complex k-space as a real-stacked even-channel tensor
        and the validation-image logger receives the model's prediction
        directly, the path through ``kspace_to_image`` can pair
        ``(coil0, coil1)`` real magnitudes as complex real/imag and IFFT
        the result — producing the doubled-brain + k-space-superposition
        artefact reported in experiment_11.

        We require that any k-space-input model whose validation
        pipeline saves images either (a) sets
        ``validation.scoring.enable_image_metrics: true`` (forces an image-domain
        comparison upstream), or (b) declares
        ``logging.save_validation_images: false``.
        """
        check_name = "validation_image_domain_safe"
        model = getattr(config, "model", None)
        logging_cfg = getattr(config, "logging", None)
        if model is None or logging_cfg is None:
            return HealthCheckResult(True, check_name, "no model/logging block", "info")
        input_type = (getattr(model, "input_type", "") or "").lower()
        if input_type not in ("kspace", "complex", "kspace_complex"):
            return HealthCheckResult(True, check_name, "model is image-domain", "info")
        save_images = bool(logging_cfg.images.save_validation if logging_cfg else False)
        # `compute_image_metrics` lives on `validation:` and NEVER existed on
        # `logging:` -- so this getattr returned False for every config ever
        # checked, making the error UNSATISFIABLE: any k-space model with
        # save_validation_images:true failed, and the remediation named a key
        # that does not exist (and `logging:` is extra="ignore", so writing it
        # did nothing). Eight arms declare `logging.compute_image_metrics: true`,
        # visibly trying to satisfy exactly this. Issue #679.
        _val = getattr(config, "validation", None)
        compute_image_metrics = bool(
            _val.scoring.enable_image_metrics if _val is not None else False
        )
        if save_images and not compute_image_metrics:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"model.input_type='{input_type}' with "
                    f"logging.save_validation_images=true but "
                    f"validation.scoring.enable_image_metrics=false. The validation "
                    f"image logger will receive raw k-space tensors and may "
                    f"misroute multi-coil real-stacked outputs through the "
                    f"complex-pair-IFFT branch, producing a doubled-brain + "
                    f"k-space-superposition image (regression of "
                    f"experiment_11_kspace_cold_diffusion). Set "
                    f"validation.scoring.enable_image_metrics=true (the flat "
                    f"spelling validation.compute_image_metrics still loads)."
                ),
                "error",
            )
        return HealthCheckResult(
            True,
            check_name,
            "validation image path is domain-safe",
            "info",
        )

    # ──────────────────────────────────────────────────────────────────
    # Phase 10 (2026-05-05): Normalization correctness audits.
    # Added after the silent-correctness audit identified the fact that
    # phase-unsafe normalization strategies (z-score, min-max) are
    # permitted by the schema with no domain-aware guard, and that the
    # mutex between ``normalize_kspace`` and image-domain normalizers
    # is enforced by a silent INFO log rather than a config-time error.
    # See docs/audits/2026-05-05-silent-correctness-audit.md §F.
    # ──────────────────────────────────────────────────────────────────

    # Image-domain normalization strategies that destroy phase when
    # applied to k-space (real-stacked or complex). Min-max rescales
    # per-tensor extrema, z-score subtracts a non-zero mean from real
    # and imag independently — both invalidate the FFT contract.
    _PHASE_UNSAFE_IMAGE_NORMS: frozenset[str] = frozenset({"standard", "minmax"})

    # Coil-processing modes that collapse complex k-space to a real-valued
    # 1-channel magnitude image *before* normalization runs — for these,
    # image-domain normalizers (minmax / z-score) are phase-safe because
    # there's no phase left to destroy.
    _COIL_MODES_PRODUCE_MAGNITUDE: frozenset[str] = frozenset({"rss_image", "magnitude"})

    def check_normalization_kspace_compatibility(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Reject phase-unsafe normalization on k-space datasets.

        Fires only when a k-space dataset retains complex/real-stacked
        layout *into* the normalizer. If ``coil_processing_mode`` is a
        magnitude-producing mode (``rss_image``, ``magnitude``), the
        image normalizer sees a 1-channel real magnitude and minmax /
        z-score are fine — no phase to destroy.
        """
        check_name = "normalization_kspace_compatibility"
        data = getattr(config, "data", None)
        if data is None:
            return HealthCheckResult(True, check_name, "no data block", "info")
        ds_type = (getattr(data, "dataset_type", "") or "").lower()
        norm_type = (data.processing.normalization_type or "").lower()
        normalize_kspace = bool(data.processing.enable_kspace_normalization)
        coil_mode = (getattr(getattr(data, "coils", None), "processing_mode", "") or "").lower()
        # Only relevant for k-space-loading datasets and only when the
        # mutex gate isn't already disabling the image normalizer.
        is_kspace_dataset = (
            # ``fastmri_kspace`` folds to ``kspace`` and ``graph_mri`` was
            # removed as a canonical type, so neither could ever arrive here;
            # the substring test below already covers bart_kspace /
            # ismrmrd_kspace.
            ds_type in {"kspace", "m4raw"} or "kspace" in ds_type
        )
        if not is_kspace_dataset or normalize_kspace:
            return HealthCheckResult(True, check_name, "n/a", "info")
        if coil_mode in self._COIL_MODES_PRODUCE_MAGNITUDE:
            return HealthCheckResult(
                True,
                check_name,
                (
                    f"coil_processing_mode={coil_mode!r} produces 1-channel "
                    "magnitude before the normalizer; minmax/z-score are "
                    "phase-safe in this configuration."
                ),
                "info",
            )
        if norm_type in self._PHASE_UNSAFE_IMAGE_NORMS:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"data.dataset_type={ds_type!r} (k-space) with "
                    f"coil_processing_mode={coil_mode!r} retains complex / "
                    f"real-stacked layout, and "
                    f"data.processing.normalization_type={norm_type!r} destroys phase: "
                    f"image-domain normalizers (z-score / min-max) compute "
                    f"per-channel mean/extrema, which mix real and imag "
                    f"of k-space differently and break the FFT contract."
                ),
                "error",
                category="normalization_kspace_compatibility",
                yaml_keys=[
                    "data.dataset_type",
                    "data.processing.normalization_type",
                    "data.coil_processing_mode",
                ],
                fix_hint=(
                    "Set data.processing.enable_kspace_normalization=true and "
                    "data.processing.normalization_type='percentile' for phase-safe "
                    "scaling, OR change coil_processing_mode to "
                    "'rss_image'/'magnitude' if the model wants 1-channel "
                    "magnitude input, OR set normalization_type='percentile'/'none'."
                ),
            )
        return HealthCheckResult(
            True,
            check_name,
            "k-space normalization choice is phase-safe",
            "info",
        )

    # Percentile-family normalizers are conceptually compatible with
    # k-space normalization — both compute a magnitude-percentile scale.
    # Setting both is redundant but not phase-destroying. Reserve the
    # mutex error for image-domain normalizers (minmax / z-score) which
    # would compete with the k-space normalizer if it ever ran.
    _PERCENTILE_NORMALIZERS: frozenset[str] = frozenset(
        {"percentile", "robust_percentile", "scalar"}
    )

    def check_normalize_kspace_image_norm_mutex(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Flag the silent mutex skip between image norm and k-space norm.

        ``torchio_transform_builder.py`` skips the image normalization
        chain with an INFO log when ``normalize_kspace=True`` and
        ``normalization_type != "none"``. Per CLAUDE.md #9 this should
        be a config-time signal, not a silent runtime skip.

        Severity tiering:
          * ``percentile``/``robust_percentile``/``scalar`` set with
            ``normalize_kspace=True``: redundant but consistent. Info.
          * ``minmax``/``standard`` with ``normalize_kspace=True``:
            ambiguous — image norm intent silently lost. Warning.
        """
        check_name = "normalize_kspace_image_norm_mutex"
        data = getattr(config, "data", None)
        if data is None:
            return HealthCheckResult(True, check_name, "no data block", "info")
        norm_type = (data.processing.normalization_type or "").lower()
        normalize_kspace = bool(data.processing.enable_kspace_normalization)
        if not normalize_kspace or norm_type in ("", "none"):
            return HealthCheckResult(True, check_name, "no mutex conflict", "info")
        if norm_type in self._PERCENTILE_NORMALIZERS:
            return HealthCheckResult(
                True,
                check_name,
                (
                    f"data.processing.enable_kspace_normalization=true and "
                    f"data.processing.normalization_type={norm_type!r} are redundant "
                    "but consistent (both percentile-based)."
                ),
                "info",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                f"data.processing.enable_kspace_normalization=true and "
                f"data.processing.normalization_type={norm_type!r} silently skip the "
                f"image normalizer (torchio_transform_builder INFO log). "
                f"The image-norm intent is lost — keep only one."
            ),
            "warning",
            category="normalize_kspace_image_norm_mutex",
            yaml_keys=[
                "data.processing.enable_kspace_normalization",
                "data.processing.normalization_type",
            ],
            fix_hint=(
                "Either drop data.processing.normalization_type (set to 'none') "
                "or set data.processing.enable_kspace_normalization=false and pick the right "
                "image normalization."
            ),
        )

    def check_coil_flatten_image_norm_trap(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Flag the real-stacked-after-flatten + image-norm trap.

        ``coil_processing_mode='flatten'`` produces a real-stacked
        ``(2C, H, W, D)`` tensor. The ``_ComplexSafeIntensityTransform``
        wrapper only checks ``is_complex()``, which is False here —
        so a downstream z-score or min-max would silently apply per-
        channel statistics to k-space real/imag streams independently,
        destroying phase. The mutex gate in §A-2 only kicks in when
        ``normalize_kspace=True``.
        """
        check_name = "coil_flatten_image_norm_trap"
        data = getattr(config, "data", None)
        if data is None:
            return HealthCheckResult(True, check_name, "no data block", "info")
        coil_mode = (getattr(getattr(data, "coils", None), "processing_mode", "") or "").lower()
        norm_type = (data.processing.normalization_type or "").lower()
        normalize_kspace = bool(data.processing.enable_kspace_normalization)
        if (
            coil_mode == "flatten"
            and not normalize_kspace
            and norm_type in self._PHASE_UNSAFE_IMAGE_NORMS
        ):
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"coil_processing_mode='flatten' produces a real-stacked "
                    f"(2C,H,W,D) k-space tensor, and "
                    f"normalization_type={norm_type!r} would apply per-channel "
                    f"image normalization independently to real and imag. "
                    f"Phase will be silently destroyed."
                ),
                "error",
                category="coil_flatten_image_norm_trap",
                yaml_keys=[
                    "data.coil_processing_mode",
                    "data.processing.normalization_type",
                    "data.processing.enable_kspace_normalization",
                ],
                fix_hint=(
                    "Set normalize_kspace=true with normalization_type='percentile', "
                    "or change coil_processing_mode (e.g. 'rss_image' for image-domain), "
                    "or set normalization_type='none'."
                ),
            )
        return HealthCheckResult(True, check_name, "no flatten+image-norm trap", "info")

    def check_kspace_percentile_range(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Flag percentile values outside (0, 1] — common percent/fraction confusion.

        The k-space normalizer expects a fraction in (0, 1] (e.g. 0.99
        for the 99th percentile). Users sometimes pass 99 expecting
        a percent; ``torch.quantile`` then errors at runtime, but
        catching it at config time saves an aborted training launch.
        """
        check_name = "kspace_percentile_range"
        data = getattr(config, "data", None)
        if data is None:
            return HealthCheckResult(True, check_name, "no data block", "info")
        kp = data.processing.kspace_percentile
        nk = data.processing.normalization_kwargs or {}
        candidates: list[tuple[str, float]] = []
        if kp is not None:
            candidates.append(("data.processing.kspace_percentile", float(kp)))
        if isinstance(nk, dict) and "percentile" in nk:
            try:
                candidates.append(
                    (
                        "data.processing.normalization_kwargs.percentile",
                        float(nk["percentile"]),
                    )
                )
            except (TypeError, ValueError):
                pass
        for key, value in candidates:
            if not (0.0 < value <= 1.0):
                return HealthCheckResult(
                    False,
                    check_name,
                    (
                        f"{key}={value} is outside (0, 1]. The k-space "
                        f"normalizer expects a quantile fraction in (0, 1] "
                        f"(e.g. 0.99 for the 99th percentile). "
                        f"Did you mean {value / 100.0:g}?"
                    ),
                    "error",
                    category="kspace_percentile_range",
                    yaml_keys=[key],
                    fix_hint=f"Set {key} to a value in (0, 1] (e.g. 0.99).",
                )
        return HealthCheckResult(
            True,
            check_name,
            "k-space percentile values are in (0, 1]",
            "info",
        )

    def check_pin_memory_no_cuda(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Flag pin_memory=True declared on a CPU-only run.

        PyTorch warns and wastes pinned host memory when ``pin_memory``
        is enabled without an accelerator. This check looks at the
        config's declared device — actual CUDA availability is a
        runtime concern, but a config explicitly setting ``device=cpu``
        with pin_memory enabled is a clear misconfiguration.
        """
        check_name = "pin_memory_no_cuda"
        data = getattr(config, "data", None)
        # `run.device` since phase 4b. The old `config.device` read resolved to
        # "" for every config, so the `device != "cpu"` guard below short-circuited
        # to a passing "n/a" and this check never ran -- while line 6651 already
        # read the CANONICAL `data.loader.pin_memory`. Half the check was migrated.
        _run = getattr(config, "run", None)
        device = (getattr(_run, "device", None) or getattr(config, "device", "") or "").lower()
        if data is None or device != "cpu":
            return HealthCheckResult(True, check_name, "n/a", "info")
        pin_memory = getattr(getattr(data, "loader", None), "pin_memory", None)
        if pin_memory is True:
            return HealthCheckResult(
                False,
                check_name,
                (
                    "data.pin_memory=true with device='cpu'. "
                    "pin_memory is a no-op on CPU and emits a PyTorch warning."
                ),
                "warning",
                category="pin_memory_no_cuda",
                yaml_keys=["data.pin_memory", "device"],
                fix_hint="Either remove data.pin_memory or set device='cuda'.",
            )
        return HealthCheckResult(True, check_name, "no pin_memory/device conflict", "info")

    # ──────────────────────────────────────────────────────────────────
    # Phase 11 (2026-05-05 round 2): Schema cross-field validators that
    # were missing from the Pydantic models. See
    # docs/audits/2026-05-05-silent-correctness-audit.md §S.
    # ──────────────────────────────────────────────────────────────────

    def check_acceleration_bounds(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Reject ``base_acceleration > max_acceleration``.

        The acceleration schedule interpolates between
        ``base_acceleration`` and ``max_acceleration``; an inverted
        range silently flips the sign of the schedule slope. The
        Pydantic schema validates each field independently but never
        compares them.
        """
        check_name = "acceleration_bounds"
        accel = getattr(config, "undersampling", None)
        if accel is None:
            return HealthCheckResult(True, check_name, "no acceleration block", "info")
        base = getattr(accel, "base_acceleration", None)
        mx = getattr(accel, "max_acceleration", None)
        if base is None or mx is None:
            return HealthCheckResult(
                True,
                check_name,
                "base or max acceleration not set",
                "info",
            )
        try:
            base_f, mx_f = float(base), float(mx)
        except (TypeError, ValueError):
            return HealthCheckResult(True, check_name, "non-numeric values", "info")
        if base_f > mx_f:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"acceleration.base_acceleration={base_f} > "
                    f"acceleration.max_acceleration={mx_f}: the schedule slope "
                    "is silently inverted (interpolation goes from base→max)."
                ),
                "error",
                category="acceleration_bounds",
                yaml_keys=[
                    "acceleration.base_acceleration",
                    "acceleration.max_acceleration",
                ],
                fix_hint=("Swap the values, or set max_acceleration >= base_acceleration."),
            )
        return HealthCheckResult(
            True,
            check_name,
            "acceleration bounds are consistent",
            "info",
        )

    def check_strategy_class_resolves(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Verify ``training.strategy_class`` resolves to a registered strategy.

        ``TrainingSettings.from_yaml`` validates ``model.model_type``
        against the model registry but does NOT validate the strategy
        path. A typo silently passes Pydantic validation and crashes
        at runtime when the factory tries to import the class.
        """
        check_name = "strategy_class_resolves"
        training = getattr(config, "training", None)
        if training is None:
            return HealthCheckResult(True, check_name, "no training block", "info")
        strategy = getattr(training, "strategy_class", None) or ""
        if not strategy:
            return HealthCheckResult(
                True,
                check_name,
                "training.strategy_class not set (legacy mode)",
                "info",
            )
        try:
            from spectramr.infrastructure.training.strategy_factory import (
                TrainingStrategyFactory,
            )

            class_paths = getattr(TrainingStrategyFactory, "STRATEGY_CLASS_PATHS", {})
        except Exception as exc:
            return HealthCheckResult(
                True,
                check_name,
                f"strategy_factory import failed; cannot validate ({exc})",
                "info",
            )
        # Two valid forms:
        #   (a) short alias registered in STRATEGY_CLASS_PATHS
        #   (b) full dotted import path "x.y.Z" — accept on syntactic
        #       grounds (importlib resolution is verified at strategy
        #       construction; doing it here would be expensive).
        if strategy in class_paths:
            return HealthCheckResult(True, check_name, "alias resolves", "info")
        if "." in strategy and not strategy.endswith("."):
            return HealthCheckResult(
                True,
                check_name,
                "fully-qualified strategy path (deferred to runtime import)",
                "info",
            )
        # Bare token that's not a registered alias is almost certainly a typo.
        return HealthCheckResult(
            False,
            check_name,
            (
                f"training.strategy_class={strategy!r} is neither a registered "
                f"alias nor a dotted import path. Registered aliases include: "
                f"{sorted(class_paths.keys())[:8]}…"
            ),
            "error",
            category="strategy_class_resolves",
            yaml_keys=["training.strategy_class"],
            fix_hint=(
                "Use a registered alias (e.g. 'reconstruction', 'diffusion') "
                "or a fully-qualified import path."
            ),
        )

    # ──────────────────────────────────────────────────────────────────── #
    # F4 / E20 — AdversarialMixin requires losses.gan at audit-time.
    #
    # The 2026-05-16 smoke audit found 35 runtime ValueError hits from
    # `AdversarialMixin.__init__` raising "AdversarialMixin requires
    # `config.losses.gan` to be set". Strategies whose class hierarchy
    # includes AdversarialMixin (training_mode in {gan, disentangled,
    # guided_sr}) need `losses.gan:` populated in the YAML. The previous
    # behavior pushed detection to strategy construction, after the
    # smoke wrapper had already invested seconds in dataloader / model
    # setup. Hoisting to audit-time per CLAUDE.md pitfall #10.
    # ──────────────────────────────────────────────────────────────────── #

    # Training modes whose resolved strategy class inherits from
    # AdversarialMixin. Sourced from
    # `src/infrastructure/training/strategy_factory.py` and confirmed by
    # `grep AdversarialMixin src/infrastructure/training/strategies/`.
    # Add new entries here when a new strategy mixes in AdversarialMixin.
    _ADVERSARIAL_TRAINING_MODES = frozenset({"gan", "disentangled", "guided_sr"})

    def check_adversarial_strategy_requires_gan_loss(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Strategies using ``AdversarialMixin`` must declare ``losses.gan``.

        Audit anchor: TODO/audit/smoke_audit_20260516.md §F4. The mixin
        raises at ``__init__`` time when ``config.losses.gan`` is None;
        promoting this to an audit-time error lets the smoke wrapper
        gate the run before it spins up the data loaders.
        """
        check_name = "adversarial_strategy_requires_gan_loss"
        training = getattr(config, "training", None)
        if training is None:
            return HealthCheckResult(
                True,
                check_name,
                "no training block",
                "info",
            )
        mode = (getattr(training, "training_mode", "") or "").lower()
        if mode not in self._ADVERSARIAL_TRAINING_MODES:
            return HealthCheckResult(
                True,
                check_name,
                (
                    f"training_mode={mode!r} does not use AdversarialMixin "
                    f"(checked set: {sorted(self._ADVERSARIAL_TRAINING_MODES)})."
                ),
                "info",
            )
        losses = getattr(config, "losses", None)
        gan = getattr(losses, "gan", None) if losses is not None else None
        if gan is None:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"training.training_mode={mode!r} resolves to a strategy "
                    "using AdversarialMixin, which requires "
                    "`config.losses.gan` to be populated. The runtime "
                    "raises ValueError('AdversarialMixin requires "
                    "`config.losses.gan` to be set') during strategy "
                    "construction."
                ),
                "error",
                category="strategy_requires_schema",
                yaml_keys=["training.training_mode", "losses.gan"],
                fix_hint=(
                    "Add a `losses.gan:` block to the YAML with at least "
                    "`lambda_adv`, `disc_updates`, `r1_interval`, "
                    "`r1_probability`. The canonical example lives in "
                    "src/spectramr/config/schemas/templates/v1.0_reference.yaml."
                ),
            )
        return HealthCheckResult(
            True,
            check_name,
            f"training_mode={mode!r}: losses.gan populated.",
            "info",
        )

    def check_themed_strategy_requires_themed_component(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Themed generic-base aliases must wire a registered themed component.

        PR-0 (2026-05-28 façade triage,
        ``TODO/pr0_facade_strategy_triage_2026_05_28.md``): keys such as
        ``tropical_mrf`` / ``koopman_fmri`` / ``sheaf_kspace`` route to a
        generic base strategy that does NOT branch on the key. Unless the
        YAML wires a *registered* loss or model that encodes the advertised
        mathematics, the run silently degrades to a vanilla baseline —
        CLAUDE.md pitfall #9. The SSOT requirement map lives in
        ``themed_strategy_components.THEMED_GENERIC_COMPONENTS``.
        """
        from spectramr.infrastructure.validation.themed_strategy_components import (
            themed_component_status,
        )

        check_name = "themed_strategy_requires_themed_component"
        training = getattr(config, "training", None)
        key = (getattr(training, "training_mode", "") or "") if training else ""

        declared: set[str] = set()
        losses_cfg = getattr(config, "losses", None)
        if losses_cfg is not None:
            for attr in LOSS_LIST_DOMAINS:
                for entry in getattr(losses_cfg, attr, None) or []:
                    if getattr(entry, "enabled", True) is False:
                        continue
                    name = (
                        getattr(entry, "name", None)
                        or getattr(entry, "loss_type", None)
                        or getattr(entry, "type", None)
                    )
                    if name:
                        declared.add(str(name))
        model_type = getattr(getattr(config, "model", None), "model_type", None)

        status = themed_component_status(key, declared, model_type)
        if not status.applicable:
            return HealthCheckResult(
                True,
                check_name,
                f"training_mode={key!r} is not a themed generic-base alias.",
                "info",
            )
        if status.satisfied:
            return HealthCheckResult(
                True,
                check_name,
                f"training_mode={key!r} wires themed component(s): {list(status.matched)}.",
                "info",
            )
        want_l = sorted(status.required.get("losses", set()))
        want_m = sorted(status.required.get("models", set()))
        return HealthCheckResult(
            False,
            check_name,
            (
                f"training_mode={key!r} routes to a generic base strategy but "
                "the config wires none of its paradigm-specific components, so "
                "it silently runs as a vanilla baseline (pitfall #9). "
                f"model_type={model_type!r}, declared_losses={sorted(declared)}."
            ),
            "error",
            category="themed_strategy_requires_themed_component",
            yaml_keys=["training.training_mode", "losses", "model.model_type"],
            fix_hint=(
                f"Wire at least one required loss {want_l} or set model_type to "
                f"one of {want_m}. Otherwise reclassify the key as an honest "
                "reconstruction/diffusion baseline (drop the novelty claim)."
            ),
        )

    def check_lion_learning_rate_scale(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Lion needs a much smaller LR than AdamW, and diverges without one.

        Lion's update is a SIGN, so every parameter moves by exactly ``lr``
        regardless of gradient magnitude. An AdamW-scale learning rate therefore
        does not train worse -- it diverges. Warning rather than error because
        the useful range is a rule of thumb (roughly 3-10x smaller), not a
        constant, and an arm may legitimately be probing it.
        """
        check_name = "lion_learning_rate_scale"
        optimization = getattr(config, "optimization", None)
        optimizer_type = str(getattr(optimization.optimizer, "type", "") or "").lower()
        if optimizer_type != "lion":
            return HealthCheckResult(
                True,
                check_name,
                f"optimizer_type={optimizer_type!r} is not lion; n/a.",
                "info",
            )
        lr = float(getattr(optimization.optimizer, "learning_rate", 0.0) or 0.0)
        #: Above this, an AdamW-tuned LR has almost certainly been copied across.
        threshold = 3e-4
        if lr <= threshold:
            return HealthCheckResult(
                True, check_name, f"lion learning_rate={lr:g} is in range.", "info"
            )
        return HealthCheckResult(
            False,
            check_name,
            f"optimizer_type='lion' with learning_rate={lr:g}. Lion's update is a "
            "sign, so every parameter moves by exactly lr; an AdamW-scale rate "
            f"(> {threshold:g}) diverges rather than merely training worse.",
            "warning",
            category="lion_learning_rate_scale",
            yaml_keys=[
                "optimization.optimizer.type",
                "optimization.optimizer.learning_rate",
            ],
            fix_hint="Use roughly 3-10x smaller than the AdamW LR, with weight "
            "decay correspondingly larger.",
        )

    def check_compile_with_sharded_strategy(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Whether this strategy can be compiled at all, and where.

        The premise this check was written against is retired. It reported a
        warning for ``fsdp``/``deepspeed`` because ``ModelBuilder.compile()`` ran
        before the wrap, giving ``FSDP(torch.compile(m))``; compilation is now
        placed per strategy, so that ordering is fixed rather than warned about.

        It also missed ``ddp`` and ``dp`` entirely, which meant the one
        combination with a measurable cost -- ``DDP(compile(m))`` forfeits
        DDPOptimizer, and with it the allreduce/backward overlap -- passed
        review in silence.

        The table in ``builders/compile_placement`` is the single owner of the
        decision; this reads it so the audit and the runtime cannot disagree
        about which combinations are legal. Only a refusal errors: those are
        measured crashes, knowable statically, and they should not cost a GPU
        allocation to discover.
        """
        check_name = "compile_with_sharded_strategy"
        optimization = getattr(config, "optimization", None)
        parallel = getattr(config, "parallel", None)
        from spectramr.infrastructure.training.builders.compile_placement import (
            resolve_compile_placement,
        )

        strategy = getattr(parallel, "strategy", "none") if parallel else "none"
        # getattr on the block, not attribute access: `optimization` is None on
        # partial configs and this used to raise AttributeError out of a health
        # check, which is the one thing a health check must never do.
        compile_cfg = getattr(optimization, "compile", None)
        if not getattr(compile_cfg, "enabled", False):
            return HealthCheckResult(
                True, check_name, "optimization.compile.enabled is off; n/a.", "info"
            )

        zero_stage = getattr(getattr(parallel, "deepspeed", None), "zero_stage", None)
        try:
            placement = resolve_compile_placement(strategy, zero_stage=zero_stage)
        except ValueError as exc:  # a strategy with no declared placement
            return HealthCheckResult(
                False,
                check_name,
                str(exc),
                "error",
                category="compile_with_sharded_strategy",
                yaml_keys=["optimization.compile.enabled", "parallel.strategy"],
            )

        if placement.refused:
            return HealthCheckResult(
                False,
                check_name,
                placement.refusal,
                "error",
                category="compile_with_sharded_strategy",
                yaml_keys=[
                    "optimization.compile.enabled",
                    "parallel.strategy",
                    "parallel.deepspeed.zero_stage",
                ],
                fix_hint=(
                    "Set optimization.compile.enabled: false and use DeepCompile instead: "
                    "parallel.deepspeed.compile.enabled: true with passes: [z3] at "
                    "zero_stage 3, or [z1] at stage 1/2. Dropping to a lower zero_stage "
                    "does NOT make torch.compile the right tool here -- every sharded "
                    "stage has a DeepCompile pass."
                ),
            )
        # The one combination the table knows is broken and the optimizer
        # cannot diagnose. Where the strategy must wrap before the optimizer
        # exists (fsdp, deepspeed), compilation lands first, so
        # `_resolve_param_groups` matches its keys against `_orig_mod.<name>`
        # and raises blaming the key rather than the wrapper (#2174). Knowable
        # from the YAML alone, so it costs an audit rather than a GPU
        # allocation and a confusing traceback.
        param_groups = getattr(
            getattr(getattr(config, "optimization", None), "optimizer", None),
            "param_groups",
            None,
        )
        if placement.optimizer_sees_wrapper and param_groups:
            return HealthCheckResult(
                False,
                check_name,
                f"optimization.compile.enabled=true with parallel.strategy={strategy!r} places "
                f"compilation at {placement.stage!r}, which is BEFORE the optimizer is built -- "
                f"so the optimizer introspects a compiled module whose parameters are named "
                f"'_orig_mod.<...>'. optimizer.param_groups keys "
                f"({sorted(param_groups)}) are matched against those names and a key that "
                f"matches nothing raises, blaming the key rather than the wrapper (#2174).",
                "error",
                category="compile_with_sharded_strategy",
                yaml_keys=[
                    "optimization.compile.enabled",
                    "parallel.strategy",
                    "optimization.optimizer.param_groups",
                ],
                fix_hint=(
                    "Drop optimization.optimizer.param_groups for this arm, or run it on a "
                    "strategy that compiles after the optimizer exists (none, ddp), or set "
                    "optimization.compile.enabled: false."
                ),
            )
        if placement.advisory:
            return HealthCheckResult(
                True,
                check_name,
                placement.advisory,
                "info",
                category="compile_with_sharded_strategy",
                yaml_keys=["optimization.compile.enabled", "parallel.strategy"],
                always_report=True,
            )
        # The resolved SELECTION, not just the placement: `apply_to` decides
        # which models are compiled at all, and an arm that meant to compile its
        # generator and named nothing gets every model instead.
        selection = getattr(compile_cfg, "apply_to", None)
        applies_to = ", ".join(selection) if selection else "every model"
        return HealthCheckResult(
            True,
            check_name,
            f"compile is placed at {placement.stage!r} for strategy={strategy!r}, "
            f"applied to {applies_to}.",
            "info",
            category="compile_with_sharded_strategy",
            always_report=True,
        )

    def check_deepspeed_extra_installed(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """``strategy: 'deepspeed'`` requires the ``[deepspeed]`` extra.

        Catches the missing dependency at audit time (~100 ms) instead of after
        the whole training environment has been built on a cluster node. Uses
        ``find_spec`` so the check never pays DeepSpeed's import cost -- it
        probes CUDA and can JIT-build ops.
        """
        import importlib.util

        check_name = "deepspeed_extra_installed"
        parallel = getattr(config, "parallel", None)
        strategy = getattr(parallel, "strategy", "none") if parallel else "none"
        if strategy != "deepspeed":
            return HealthCheckResult(
                True,
                check_name,
                f"strategy={strategy!r} is not deepspeed; n/a.",
                "info",
            )
        if importlib.util.find_spec("deepspeed") is not None:
            return HealthCheckResult(True, check_name, "deepspeed is importable.", "info")
        return HealthCheckResult(
            False,
            check_name,
            "parallel.strategy='deepspeed' but the deepspeed package is not "
            "importable -- the run would fail after building the whole training "
            "environment.",
            "error",
            category="deepspeed_extra_required",
            yaml_keys=["parallel.strategy"],
            fix_hint="pip install -e '.[deepspeed]'",
        )

    def check_deepspeed_topology_coherent(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """The declared ZeRO topology must be one DeepSpeed can actually run.

        Two arrangements pass every schema check and then behave as something
        other than what the arm claims:

        * **Offload without a stage that owns the state.** ZeRO-1 partitions
          optimizer state and ZeRO-0 partitions nothing, so
          ``offload_optimizer: cpu`` at stage 0 offloads an empty partition:
          the arm advertises CPU offload as its memory story, pays none of the
          transfer cost, and gets none of the saving.
        * **Parameter offload below stage 3.** ``offload_param`` is only
          meaningful once parameters are partitioned, which is stage 3 alone.

        A warning rather than an error: the run is valid and will train, and an
        arm may be deliberately sweeping stages with a fixed offload block. The
        claim is what is wrong, and the audit's job is to say so before the
        cluster time is spent.
        """
        check_name = "deepspeed_topology_coherent"
        parallel = getattr(config, "parallel", None)
        strategy = getattr(parallel, "strategy", "none") if parallel else "none"
        if strategy != "deepspeed":
            return HealthCheckResult(
                True,
                check_name,
                f"strategy={strategy!r} is not deepspeed; n/a.",
                "info",
            )
        ds = getattr(parallel, "deepspeed", None)
        stage = getattr(ds, "zero_stage", 0) if ds else 0
        offload_optimizer = getattr(ds, "offload_optimizer", "none") if ds else "none"
        offload_param = getattr(ds, "offload_param", "none") if ds else "none"

        problems: list[str] = []
        if offload_optimizer != "none" and stage < 1:
            problems.append(
                f"offload_optimizer={offload_optimizer!r} at zero_stage={stage}: "
                "optimizer state is not partitioned below stage 1, so there is "
                "nothing to offload"
            )
        if offload_param != "none" and stage < 3:
            problems.append(
                f"offload_param={offload_param!r} at zero_stage={stage}: "
                "parameters are only partitioned at stage 3, so this is inert"
            )
        if problems:
            return HealthCheckResult(
                False,
                check_name,
                "DeepSpeed topology advertises offloading that this stage cannot "
                "perform: " + "; ".join(problems) + ".",
                "warning",
                category="deepspeed_topology_incoherent",
                yaml_keys=[
                    "parallel.deepspeed.zero_stage",
                    "parallel.deepspeed.offload_optimizer",
                    "parallel.deepspeed.offload_param",
                ],
                fix_hint=(
                    "Raise zero_stage to match the offload you want (>=1 for "
                    "optimizer, 3 for param), or set the offload back to 'none'."
                ),
            )
        return HealthCheckResult(
            True,
            check_name,
            f"zero_stage={stage} is coherent with the declared offloading.",
            "info",
        )

    def check_deepspeed_zero_stage_has_ranks_to_shard(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """ZeRO partitions ACROSS RANKS. At world size 1 it partitions nothing.

        The sibling check above asks whether the *stage* owns the state the arm
        wants offloaded. This asks the other half of the same question: whether
        there is anyone to shard it to. Every ZeRO stage divides state by world
        size, so at one rank stage 1/2/3 all reduce to plain single-process
        training -- while still paying for it. The engine builds the reduction
        buckets regardless (``reduce_bucket_size`` and ``allgather_bucket_size``
        default to 500 MB each), installs the gradient hooks, and runs the
        partitioning bookkeeping. Single-rank ZeRO is therefore not merely
        neutral: it is measurably worse than ``zero_stage: 0``, and it reports
        itself in the logs as sharding (``[DeepSpeed] engine initialised:
        zero_stage=2``) while sharding across exactly one process.

        **Advisory, never a gate, and the polarity is the whole design.** The
        resolved ``num_devices`` is almost never a declaration: it defaults to 1
        (``defaults_provider``), and ``pipelines/distributed.py`` OVERWRITES both
        it and ``num_nodes`` from ``LOCAL_WORLD_SIZE`` on any ``train-distributed``
        launch, because those are observed facts about the **launch** rather than
        user intent. A warning here would therefore fire on every launcher-driven
        DeepSpeed arm in the corpus -- including ones that shard perfectly well at
        run time -- and no config edit could silence it, since the knob the
        warning pointed at is the one the launcher discards. Reading a defaulted
        value as a declaration is exactly the substitution the no-silent-default
        rule forbids, so this returns ``passed=True`` at ``info`` severity and
        contributes to neither ``report.passed`` nor ``report.warnings``.

        Two corrections to that reasoning, both from #1275. First, ``LOCAL_WORLD_SIZE``
        is what ``torchrun --nproc_per_node`` was *given*; it is not the size of the
        scheduler's grant, and the gap between the two is the whole of #1274 (a
        ``--gpus=4`` allocation running one rank reaches this check reporting a
        world size of 1 and is right to). Second, the "defaulted, not declared"
        rationale is a statement about the *audit* path: by the time ``train.py``
        runs this check the value has already been overwritten, so there it is a
        measured launch fact. The polarity still holds -- a measured 1 is not a
        config error either, and there is no YAML edit that answers it -- but it
        holds for a different reason on each surface.

        Visible without gating: the finding branch sets ``always_report``, so
        ``log_summary`` emits it on the train path. Without it the message existed
        only for ``audit``, which renders passing results; ``log_summary`` skips
        them, and the incident this check was written for printed
        ``Config Health: 141/141 checks passed`` and not one word of the diagnosis.

        What it is good for is the path where the declaration IS the world size:
        ``launch_distributed`` runs the arm inline when ``num_devices == 1 and
        num_nodes == 1`` (``infrastructure/distributed/launcher.py``), so a
        campaign-launched arm at this topology really does get inert ZeRO.

        The guidance lives in the message rather than ``fix_hint`` because
        ``__rich__`` renders a hint only on a non-passing result; on a passing
        one it would reach ``--json`` and nothing else.
        """
        check_name = "deepspeed_zero_stage_has_ranks_to_shard"
        parallel = getattr(config, "parallel", None)
        strategy = getattr(parallel, "strategy", "none") if parallel else "none"
        if strategy != "deepspeed":
            return HealthCheckResult(
                True,
                check_name,
                f"strategy={strategy!r} is not deepspeed; n/a.",
                "info",
            )
        ds = getattr(parallel, "deepspeed", None)
        stage = getattr(ds, "zero_stage", 0) if ds else 0
        if stage == 0:
            return HealthCheckResult(
                True,
                check_name,
                "zero_stage=0 partitions nothing by design; n/a.",
                "info",
            )

        num_devices = getattr(parallel, "num_devices", 1) or 1
        num_nodes = getattr(parallel, "num_nodes", 1) or 1
        declared_world = num_devices * num_nodes
        if declared_world > 1:
            return HealthCheckResult(
                True,
                check_name,
                f"zero_stage={stage} has {declared_world} declared rank(s) "
                f"({num_devices} device(s) x {num_nodes} node(s)) to partition "
                "across.",
                "info",
            )

        return HealthCheckResult(
            True,
            check_name,
            f"zero_stage={stage} at a DECLARED world size of 1 "
            f"({num_devices} device(s) x {num_nodes} node(s)) partitions nothing: "
            "ZeRO divides optimizer state, gradients and parameters across ranks, "
            "so at one rank every stage reduces to single-process training while "
            "still building the reduction buckets and gradient hooks. Advisory "
            "only, because `train-distributed` overwrites num_devices from the "
            "launch (LOCAL_WORLD_SIZE -- what torchrun --nproc_per_node was "
            "given, NOT the size of the scheduler's grant) -- launched under "
            "torchrun with >=2 ranks this arm shards normally, and nothing here "
            "needs changing. It matters on the campaign path, where "
            "`launch_distributed` runs a 1x1 arm inline and the declaration IS "
            "the world size: there, launch with >=2 ranks or set zero_stage: 0 "
            "to say single-rank on purpose.",
            "info",
            category="deepspeed_zero_stage_inert_at_single_rank",
            yaml_keys=[
                "parallel.deepspeed.zero_stage",
                "parallel.num_devices",
                "parallel.num_nodes",
            ],
            fix_hint=(
                "Launch with >=2 ranks (sbatch --gpus=<type>:N -> torchrun), or "
                "declare zero_stage: 0 for a deliberate single-rank run."
            ),
            # The one branch that states a finding rather than an n/a. Its three
            # siblings above stay unflagged: "not deepspeed", "zero_stage=0" and
            # "has N ranks to partition across" are confirmations, and emitting
            # those is how an advisory channel becomes noise nobody reads.
            always_report=True,
        )

    def check_zenflow_accumulation_conflict(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """ZenFlow OVERWRITES ``gradient_accumulation_steps``. Error on a conflict.

        ``configure_zenflow`` ends with::

            engine._config.gradient_accumulation_steps = engine.update_interval

        So an arm declaring ``optimization.gradient.accumulation_steps: 4`` and a
        ZenFlow ``update_interval: 16`` trains at an effective batch **4x larger
        than its config states**, while every artifact -- provenance, the
        run banner, ``effective_batch_size`` -- reports 4. That is a successful
        run of a different experiment, which is precisely the failure the
        no-``config_path`` design exists to prevent; it would be perverse to
        close that door and leave this one open.

        An error, not a warning: nothing downstream can detect the substitution,
        and two arms differing only in ``update_interval`` are not comparable
        even though their declared batch sizes match.
        """
        check_name = "zenflow_accumulation_conflict"
        parallel = getattr(config, "parallel", None)
        strategy = getattr(parallel, "strategy", "none") if parallel else "none"
        ds = getattr(parallel, "deepspeed", None) if parallel else None
        zf = getattr(ds, "zenflow", None)
        if strategy != "deepspeed" or zf is None or not zf.enabled:
            return HealthCheckResult(True, check_name, "ZenFlow not enabled; n/a.", "info")

        declared = getattr(
            getattr(getattr(config, "optimization", None), "gradient", None),
            "accumulation_steps",
            1,
        )
        declared = int(declared or 1)
        update_interval = zf.update_interval

        if not isinstance(update_interval, int):
            # "auto" -> engine.update_interval = 0 -> accumulation becomes 0.
            if declared > 1:
                return HealthCheckResult(
                    False,
                    check_name,
                    f"ZenFlow update_interval='auto' makes DeepSpeed set "
                    f"gradient_accumulation_steps from its own scheduler, "
                    f"discarding the declared value of {declared}. The run's "
                    "effective batch size will not be the one this config states.",
                    "error",
                    category="zenflow_overwrites_accumulation",
                    yaml_keys=[
                        "optimization.gradient.accumulation_steps",
                        "parallel.deepspeed.zenflow.update_interval",
                    ],
                    fix_hint=(
                        "Set optimization.gradient.accumulation_steps: 1, or pin "
                        "zenflow.update_interval to the integer you want."
                    ),
                )
            return HealthCheckResult(
                True,
                check_name,
                "ZenFlow owns the accumulation schedule and no conflicting value is declared.",
                "info",
            )

        if update_interval != declared:
            return HealthCheckResult(
                False,
                check_name,
                f"optimization.gradient.accumulation_steps={declared} but ZenFlow "
                f"update_interval={update_interval} REPLACES it at engine setup. "
                f"The run would accumulate over {update_interval} micro-batches "
                f"while every artifact reports {declared}.",
                "error",
                category="zenflow_overwrites_accumulation",
                yaml_keys=[
                    "optimization.gradient.accumulation_steps",
                    "parallel.deepspeed.zenflow.update_interval",
                ],
                fix_hint=(
                    f"Make the two agree: set "
                    f"optimization.gradient.accumulation_steps: {update_interval}."
                ),
            )
        return HealthCheckResult(
            True,
            check_name,
            f"gradient_accumulation_steps and ZenFlow update_interval agree ({declared}).",
            "info",
        )

    def check_deepcompile_supported(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """DeepCompile needs torch>=2.6 on CUDA, and conflicts with torch.compile.

        Two failures, both of which otherwise surface only as reduced throughput:

        * **Unsupported stack.** ``is_deepcompile_supported()`` gates on torch
          version and accelerator; below it the engine simply does not apply the
          passes.
        * **Stacked with ``optimization.compile.enabled``.** ``ModelBuilder.compile``
          runs ``torch.compile`` in Stage A, *before* ``deepspeed.initialize``,
          so DeepCompile would be asked to insert ZeRO passes into a graph it
          never traced. The two are alternatives, not layers -- DeepCompile
          exists precisely because compiling the module first leaves the
          collectives opaque.
        """
        check_name = "deepcompile_supported"
        parallel = getattr(config, "parallel", None)
        strategy = getattr(parallel, "strategy", "none") if parallel else "none"
        ds = getattr(parallel, "deepspeed", None) if parallel else None
        compile_cfg = getattr(ds, "compile", None)
        if strategy != "deepspeed" or compile_cfg is None or not compile_cfg.enabled:
            return HealthCheckResult(True, check_name, "DeepCompile not enabled; n/a.", "info")

        if getattr(
            getattr(getattr(config, "optimization", None), "compile", None),
            "enabled",
            False,
        ):
            return HealthCheckResult(
                False,
                check_name,
                "optimization.compile.enabled and "
                "parallel.deepspeed.compile.enabled are both set. torch.compile "
                "runs BEFORE deepspeed.initialize, so DeepCompile would receive "
                "an already-compiled module and could not rewrite the ZeRO "
                "collectives -- which is the only thing it is for.",
                "error",
                category="deepcompile_conflicts_with_torch_compile",
                yaml_keys=[
                    "optimization.compile.enabled",
                    "parallel.deepspeed.compile.enabled",
                ],
                fix_hint="Set optimization.compile.enabled: false; DeepCompile subsumes it.",
            )

        try:
            from deepspeed.compile.util import is_deepcompile_supported

            supported = bool(is_deepcompile_supported())
        except Exception:
            # The extra's absence is check_deepspeed_extra_installed's job; not
            # answering twice keeps one failure to one message.
            return HealthCheckResult(
                True,
                check_name,
                "deepspeed not importable here; support probe skipped (see "
                "deepspeed_extra_installed).",
                "info",
            )

        if not supported:
            return HealthCheckResult(
                False,
                check_name,
                "DeepCompile is enabled but is_deepcompile_supported() is False "
                "on this host (it requires torch>=2.6 and a CUDA accelerator). "
                "The passes would not be applied.",
                "error",
                category="deepcompile_unsupported_stack",
                yaml_keys=["parallel.deepspeed.compile.enabled"],
                fix_hint="Upgrade torch to >=2.6 on a CUDA host, or set "
                "parallel.deepspeed.compile.enabled: false.",
            )
        return HealthCheckResult(
            True, check_name, "DeepCompile is supported on this stack.", "info"
        )

    def check_optimizer_registered(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """``optimization.optimizer.type`` must resolve in the optimizer registry.

        The enum and the registry are asserted in lockstep at import, so this
        normally passes trivially. It earns its place on the ladder because the
        failure it catches is *misdiagnosed* otherwise: a
        ``@register_optimizer`` in a module nothing imports is dead, and the
        symptom is the name vanishing from ``list_available()``. The runtime
        then reports "unknown optimizer", which reads as a typo in the YAML and
        sends the user to edit a config that was correct all along.
        """
        check_name = "optimizer_registered"
        optimization = getattr(config, "optimization", None)
        declared = getattr(optimization.optimizer, "type", None)
        if not declared:
            return HealthCheckResult(True, check_name, "no optimizer_type declared; n/a.", "info")
        try:
            from spectramr.infrastructure.training.optimizer_registry import (
                OptimizerRegistry,
            )

            available = set(OptimizerRegistry.list_available())
        except Exception as exc:  # pragma: no cover - import-time registry break
            return HealthCheckResult(
                False,
                check_name,
                f"optimizer registry could not be imported: {exc}",
                "error",
                category="optimizer_registry_broken",
                yaml_keys=["optimization.optimizer.type"],
            )
        name = str(declared).strip().lower()
        if name in available:
            return HealthCheckResult(
                True, check_name, f"optimizer_type={name!r} is registered.", "info"
            )
        return HealthCheckResult(
            False,
            check_name,
            f"optimizer_type={name!r} is not registered. Available: {sorted(available)}.",
            "error",
            category="optimizer_not_registered",
            yaml_keys=["optimization.optimizer.type"],
            fix_hint=(
                "If the name looks correct, this is an IMPORT problem, not a "
                "typo: the @register_optimizer decorator lives in a module that "
                "nothing imports. Check "
                "infrastructure/training/optimizers/__init__.py."
            ),
        )

    def check_deepspeed_precision_coherent(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """DeepSpeed fp16 is incompatible with complex/k-space arms.

        ``get_autocast_context`` deliberately disables autocast for complex+fp16
        because there is no ``complex16``. DeepSpeed fp16 casts module weights to
        half *from inside the engine*, where that guard cannot see it -- so the
        one arm class the guard exists for is exactly the one it stops
        protecting.
        """
        check_name = "deepspeed_precision_coherent"
        parallel = getattr(config, "parallel", None)
        strategy = getattr(parallel, "strategy", "none") if parallel else "none"
        if strategy != "deepspeed":
            return HealthCheckResult(
                True,
                check_name,
                f"strategy={strategy!r} is not deepspeed; n/a.",
                "info",
            )

        optimization = getattr(config, "optimization", None)
        use_amp = bool(getattr(optimization.precision, "enabled", False))
        amp_dtype = getattr(optimization.precision, "dtype", None)
        is_fp16 = use_amp and amp_dtype in (None, "float16")
        if not is_fp16:
            return HealthCheckResult(
                True, check_name, f"amp_dtype={amp_dtype!r} is not fp16; n/a.", "info"
            )

        model = getattr(config, "model", None)
        target_domain = str(getattr(model, "target_domain", "") or "").lower()
        physics = getattr(config, "physics", None)
        kspace = getattr(physics, "kspace", None) if physics else None
        kspace_recon = bool(getattr(kspace, "enable_kspace_recon", False))
        if "kspace" not in target_domain and not kspace_recon:
            return HealthCheckResult(
                True, check_name, "Not a complex/k-space arm; fp16 is fine.", "info"
            )

        return HealthCheckResult(
            False,
            check_name,
            "parallel.strategy='deepspeed' with fp16 on a k-space/complex arm. "
            "DeepSpeed casts weights to half inside the engine, where the "
            "complex+fp16 autocast guard cannot see it; complex64 activations "
            "against half weights produce NaNs, not a slowdown.",
            "error",
            category="deepspeed_precision_coherent",
            yaml_keys=["parallel.strategy", "optimization.precision.dtype"],
            fix_hint="Set optimization.precision.dtype: 'bfloat16' (or disable AMP).",
        )

    @staticmethod
    def _resolves_to_diffusion_strategy(config: TrainingSettings) -> bool:
        """Whether this arm's RESOLVED training strategy is a diffusion objective.

        Two signals, unioned, both derived at runtime. Neither is sufficient
        alone -- measured over all 204 ``training_mode`` keys:

        * ``issubclass(DiffusionTrainingStrategy)`` misses **15**: the whole
          cold-diffusion family plus ``flow_matching``, ``rectified_flow``,
          ``x_diffusion`` and the riemannian variants inherit straight from
          ``BaseTrainingStrategy``.
        * a ``"diffusion"`` substring on the strategy qualname misses **9**:
          ``edm`` (Elucidated Diffusion Models), ``i2sb``,
          ``stochastic_interpolants``, ``twin_dps``,
          ``bloch_schrodinger_bridge``, ``coord_kspace_gen``, ``padnet``,
          ``flow_matching_pfode``, ``vf_consistency_distillation`` -- diffusion
          objectives whose class and module names say nothing.

        A third signal was tried and **rejected**: ``training.diffusion is not
        None`` adds 14 arms, all false positives (PINN, FNO, Vision-Mamba and
        VAE reconstruction arms that merely carry the sub-block) and zero true
        ones. In a check that hard-errors, over-capture blocks AMP on arms
        legitimately entitled to it, which is worse than under-capture.

        Keying on the resolved strategy rather than a ``model_type`` substring
        is deliberate: the substring form is the defect #806 removed from
        ``AMPPolicy.should_use_amp``.

        The durable fix is a declaration, not this inference --
        ``StrategyCapabilities.supported_paradigms`` is the seam and is
        populated on 0 of 204 strategies (issue #810). Collapse this to a lookup
        when it is.
        """
        from spectramr.infrastructure.training.strategies.diffusion import (
            DiffusionTrainingStrategy,
        )
        from spectramr.infrastructure.training.strategy_factory import (
            TrainingStrategyFactory,
        )

        try:
            # get_strategy_class is an INSTANCE method, not a classmethod.
            strategy_cls = TrainingStrategyFactory().get_strategy_class(config)
        except Exception:
            # An unresolvable strategy is a different check's finding; this one
            # must not turn it into a precision error.
            return False

        if isinstance(strategy_cls, type) and issubclass(strategy_cls, DiffusionTrainingStrategy):
            return True
        qualname = f"{strategy_cls.__module__}.{strategy_cls.__name__}".lower()
        return "diffusion" in qualname

    def check_diffusion_precision_policy(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Diffusion arms must train in fp32 -- no autocast, fp16 or bf16.

        Not a hypothetical ratchet. When this check was written, **12 arms in
        ``experiments/inprogress/`` resolved AMP ON while training a diffusion
        objective** (11 fp16, 1 bf16); PR #809 is what emptied it.

        They were invisible because ``optimization.use_amp`` is a ``RENAMES``
        entry that folds onto ``optimization.precision.enabled``: a grep for the
        canonical key reports 0 while the legacy spelling is live. This check
        asks ``resolve_amp_precision`` -- the resolver ``BaseTrainingStrategy``
        and ``build_deepspeed_config`` already use -- so it sees both spellings
        and the third state (``dtype: 'float32'`` disables AMP even when
        ``enabled`` is true).

        bf16 is refused alongside fp16. bf16 is the *safer* half-precision, and
        elsewhere in this file it is the recommended fix
        (``check_deepspeed_precision_coherent``), but the policy here is fp32
        for diffusion and a check that allowed one half-precision path would
        leave the constraint half-enforced.
        """
        from spectramr.infrastructure.training.mixed_precision import (
            resolve_amp_precision,
        )

        check_name = "diffusion_precision_policy"

        if not self._resolves_to_diffusion_strategy(config):
            return HealthCheckResult(
                True,
                check_name,
                "Not a diffusion training strategy; n/a.",
                "info",
            )

        precision = config.optimization.precision
        amp_enabled, resolved = resolve_amp_precision(precision.enabled, precision.dtype)
        if not amp_enabled:
            return HealthCheckResult(
                True,
                check_name,
                f"Diffusion arm trains in fp32 (precision.enabled="
                f"{precision.enabled}, dtype={precision.dtype!r}).",
                "info",
            )

        return HealthCheckResult(
            False,
            check_name,
            f"Diffusion arm resolves AMP ON ({resolved}). The noise/score-"
            "prediction objective trains in full fp32 in this framework. Note "
            "the legacy `optimization.use_amp` folds onto "
            "`optimization.precision.enabled`, so this fires on either "
            "spelling.",
            "error",
            category="diffusion_precision_policy",
            yaml_keys=[
                "optimization.precision.enabled",
                "optimization.precision.dtype",
            ],
            fix_hint=(
                "Set optimization.precision: {enabled: false, dtype: float32}. "
                "If the arm also declares the legacy `optimization.use_amp`, "
                "REPLACE it -- the schema raises when the two disagree."
            ),
        )

    def check_bf16_requires_ampere(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """bf16 below sm_80 is emulated, and torch reports that as supported.

        ``torch.cuda.is_bf16_supported()`` defaults to ``including_emulation=True``
        and answers **True** on a V100, because that branch only checks a bf16
        tensor can be created. The target clusters are sm_70. So a gate written
        against the obvious probe would confirm bf16 on exactly the hardware it
        needed to reject; this reads the compute capability instead.

        **Severity is deliberate.** ``audit`` is ``--strict`` and warnings exit 2
        (non-negotiable 4), while the audit legitimately runs on a login node
        whose GPU differs from the compute node's, or which has none. So an
        *unverifiable* answer must not gate: it reports at ``info`` with
        ``always_report`` and names the env var that makes it decidable. Only a
        capability this machine could actually read produces an error. The
        runtime gate in ``mixed_precision.assert_amp_dtype_supported`` is the
        enforcer, and it cannot be fooled by where the audit ran.
        """
        from spectramr.core.device_capabilities import (
            TARGET_CAPABILITY_ENV,
            probe_device_capabilities,
        )
        from spectramr.infrastructure.training.mixed_precision import (
            resolve_amp_precision,
        )

        check_name = "bf16_requires_ampere"
        optimization = getattr(config, "optimization", None)
        precision = getattr(optimization, "precision", None)
        try:
            enabled, resolved = resolve_amp_precision(
                bool(getattr(precision, "enabled", False)),
                getattr(precision, "dtype", None),
            )
        except ValueError:  # the schema owns the vocabulary; it already raised
            return HealthCheckResult(True, check_name, "AMP dtype is invalid; n/a.", "info")

        parallel = getattr(config, "parallel", None)
        fsdp = getattr(parallel, "fsdp", None)
        fsdp_bf16 = bool(getattr(fsdp, "enabled", False)) and (
            getattr(fsdp, "mixed_precision", None) == "bf16"
        )
        # FSDP's sub-block defaults to bf16, so most sharded arms never declared
        # it -- but the policy still applies, and only when sharding is on.
        if not ((enabled and resolved == "bf16") or fsdp_bf16):
            return HealthCheckResult(
                True, check_name, "bf16 is not the resolved AMP dtype; n/a.", "info"
            )

        caps = probe_device_capabilities()
        declared_by = (
            "optimization.precision.dtype" if enabled and resolved == "bf16"
            else "parallel.fsdp.mixed_precision"
        )
        keys = ["optimization.precision.dtype", "parallel.fsdp.mixed_precision"]

        if caps.device_type != "cuda" or "capability" in caps.incomplete:
            return HealthCheckResult(
                True,
                check_name,
                f"bf16 declared ({declared_by}), but this machine cannot report a CUDA "
                f"compute capability (device_type={caps.device_type!r}, source={caps.source}). "
                f"Native bf16 needs sm_80+ and the run RAISES below that, so this is "
                f"unverified rather than passed. Declare the target with "
                f"{TARGET_CAPABILITY_ENV}=<major>.<minor> to decide it here.",
                "info",
                category="bf16_capability",
                yaml_keys=keys,
                always_report=True,
            )
        if caps.native_bf16:
            return HealthCheckResult(
                True,
                check_name,
                f"bf16 on sm_{caps.capability_str} (native).",
                "info",
            )
        return HealthCheckResult(
            False,
            check_name,
            f"bf16 declared ({declared_by}) on a device with compute capability "
            f"{caps.capability_str} (source={caps.source}), which has no native bf16 -- "
            f"torch would emulate it, slower than fp16 and numerically unlike the "
            f"declaration. Supported here: {list(caps.amp_dtypes)}.",
            "error",
            category="bf16_capability",
            yaml_keys=keys,
            fix_hint=(
                "Use optimization.precision.dtype: float32 for complex/k-space arms "
                "(fp16 autocast is a no-op there), or float16 otherwise. Ampere or "
                "newer is what bf16 needs."
            ),
        )

    @staticmethod
    def _complex_arm_signals(config: TrainingSettings) -> list[str]:
        """Signals that this arm carries genuine ``complex64`` tensors.

        Four, unioned; measured contribution over the 642 ``inprogress/`` arms
        (234 complex, 408 real -- the signals overlap heavily):

        ==============================  =====
        ``capabilities.input/output_domain`` in {kspace, complex_image}   180
        ``physics.kspace.enable_kspace_recon``                            133
        ``model.target_domain`` contains "kspace"                          21
        ``capabilities.accepts_complex is True``                           16
        ==============================  =====

        The last one is the *declarative* signal and would be the whole answer if
        it were populated. It is not: ``accepts_complex`` is ``None`` on **484 of
        589** registered models, ``False`` on 89, ``True`` on 16. So it is used
        as one signal among four rather than trusted alone -- the same shape as
        ``supported_paradigms`` (issue #810), and the reason this is an inference
        rather than a lookup.

        The middle two are the same signals ``check_deepspeed_precision_coherent``
        already uses to identify a complex/k-space arm; kept consistent on
        purpose so the two checks cannot disagree about what "complex" means.

        NOT a signal: using ``ComplexConv2d``. Measured -- that layer stores real
        and imaginary parts as separate REAL tensors and performs one fused real
        ``F.conv2d`` against a block weight matrix, returning ``float32``. It
        compiles cleanly under ``fullgraph=True``. "Uses complex arithmetic" and
        "carries complex dtype" are different properties, and only the second one
        is what Inductor cannot codegen.
        """
        signals: list[str] = []

        model = getattr(config, "model", None)
        capabilities = None
        try:
            from spectramr.models.init_registry import populate_model_registry
            from spectramr.models.registry import MODEL_REGISTRY

            populate_model_registry()
            entry = MODEL_REGISTRY.get(str(getattr(model, "model_type", "")).lower())
            if isinstance(entry, dict):
                capabilities = entry.get("capabilities")
        except Exception:  # pragma: no cover - registry unavailable
            capabilities = None

        if getattr(capabilities, "accepts_complex", None) is True:
            signals.append("model.capabilities.accepts_complex")

        domains: set[str] = set()
        for attr in ("input_domain", "output_domain"):
            value = getattr(capabilities, attr, None)
            if isinstance(value, str):
                domains.add(value)
            elif value:
                domains.update(str(v) for v in value)
        if domains & {"kspace", "complex_image"}:
            signals.append("model.capabilities.input/output_domain")

        target_domain = str(getattr(model, "target_domain", "") or "").lower()
        if "kspace" in target_domain:
            signals.append("model.target_domain")

        physics = getattr(config, "physics", None)
        kspace = getattr(physics, "kspace", None) if physics else None
        if bool(getattr(kspace, "enable_kspace_recon", False)):
            signals.append("physics.kspace.enable_kspace_recon")

        return signals

    def check_compile_with_complex_model(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """``torch.compile`` on a complex/k-space arm is a false throughput claim.

        This does NOT crash, and that is the whole problem. Measured on
        torch 2.11 against this repo's own ``fft2c``/``ifft2c`` round-trip at
        ``complex64``: compilation SUCCEEDS, under ``fullgraph=True``, and is
        numerically correct (max abs error 3.2e-07). Inductor simply emits

            UserWarning: Torchinductor does not support code generation for
            complex operators. Performance may be worse than eager.

        and falls back to eager for every complex operator.

        So the arm declares ``compile.enabled: true``, the audit accepts it,
        provenance stamps a compiled run, and the complex regions execute
        eagerly -- *possibly slower than not compiling at all*, since the
        guard/graph-break machinery is not free. That is pitfall #16 with a
        paper trail arguing it worked, and it is the same failure the DeepSpeed
        backend's ``_activate_deepcompile`` exists to prevent: "a run that
        silently falls back to eager is reporting throughput numbers that belong
        to a different configuration".

        Severity is ``error`` rather than ``warning`` because the declaration is
        not merely suboptimal -- it is untrue about what ran, and nothing
        downstream can tell. No arm in the corpus is affected today (234 complex
        arms, 0 with compile enabled), so this is a ratchet: it prevents the
        combination rather than fixing existing ones.

        If a future arm genuinely wants compilation for its real-valued majority
        while accepting eager complex regions, that is an owner decision to relax
        this check with a per-arm opt-out -- not something to leave implicit.
        """
        check_name = "compile_with_complex_model"

        compile_cfg = getattr(getattr(config, "optimization", None), "compile", None)
        torch_compile = bool(getattr(compile_cfg, "enabled", False))

        # DeepCompile counts. It compiles the ENGINE graph rather than the
        # module, but it hands that graph to the same code generator --
        # `deepspeed/compile/backend.py` calls `torch._inductor.compile`, and the
        # CUDA accelerator's `get_compile_backend()` is "inductor" -- so the
        # complex-op fallback is identical.
        #
        # This gate used to read `optimization.compile.enabled` alone, which made
        # it structurally unable to fire on a DeepCompile arm: the two are
        # mutually exclusive (`check_deepcompile_supported`), so a DeepCompile
        # arm has torch.compile OFF by construction and the check returned "n/a".
        # Live for the kspace_filling cohort, where 70 of 73 arms are DeepSpeed
        # ZeRO-2 with complex k-space signals.
        deepspeed_cfg = getattr(getattr(config, "parallel", None), "deepspeed", None)
        deep_compile = bool(
            getattr(getattr(deepspeed_cfg, "compile", None), "enabled", False)
        )

        if not (torch_compile or deep_compile):
            return HealthCheckResult(
                True, check_name, "no compilation is enabled; n/a.", "info"
            )
        compiler = "optimization.compile" if torch_compile else "DeepCompile"

        signals = self._complex_arm_signals(config)
        if not signals:
            return HealthCheckResult(
                True,
                check_name,
                f"Real-valued arm; {compiler} is fine.",
                "info",
            )

        if bool(getattr(compile_cfg, "allow_complex", False)):
            # The opt-out the docstring above anticipated. It is only reachable
            # WITH `regional` -- the schema refuses the pair otherwise -- because
            # it rests on the physics SSOT being fenced out of every graph
            # (`core.compile_fences`), so the compiled regions provably hold no
            # complex tensors. Measured on a real-backbone/complex-physics model:
            # unfenced 1 graph / 19 ops with the complex ops inside, fenced
            # 2 graphs / 9 ops with them excluded.
            return HealthCheckResult(
                True,
                check_name,
                "optimization.compile.allow_complex is set on a complex/k-space arm "
                f"(signals: {', '.join(signals)}). The physics ops are fenced out of "
                "the compiled graph, so only the real-valued regions compile -- at the "
                "cost of one graph break per fence. Confirm the arm is actually faster "
                "than eager before relying on it.",
                "info",
                category="compile_with_complex_model",
                yaml_keys=["optimization.compile.allow_complex"],
                always_report=True,
            )

        return HealthCheckResult(
            False,
            check_name,
            f"{compiler} is enabled on a complex/k-space arm "
            f"(signals: {', '.join(signals)}). Torchinductor cannot generate "
            "code for complex operators -- it does not fail, it falls back to "
            "eager and warns once, so the run would report a compiled "
            "configuration while executing the complex regions eagerly, "
            "possibly slower than eager throughout.",
            "error",
            category="compile_with_complex_model",
            yaml_keys=[
                "optimization.compile.enabled",
                "parallel.deepspeed.compile.enabled",
                "model.target_domain",
            ],
            fix_hint=(
                (
                    "Set optimization.compile.enabled: false. To compile the "
                    "real-valued majority anyway, set compile.regional: true and "
                    "compile.allow_complex: true, which fences the physics ops out "
                    "of the graph."
                    if torch_compile
                    else
                    # No regional opt-out here, and that is not an omission:
                    # DeepCompile compiles the whole engine graph, so there is no
                    # per-region boundary for the fences to sit on.
                    "Set parallel.deepspeed.compile.enabled: false. DeepCompile has "
                    "no regional mode -- it compiles the whole engine graph, so the "
                    "physics fences that make optimization.compile.allow_complex "
                    "honest have nowhere to apply."
                )
                + " Complex arms get their throughput from bfloat16 AMP, "
                "optimizer.fused and ZeRO instead -- see docs/training_throughput.rst."
            ),
        )

    def check_deepspeed_consolidated_best_checkpoint(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """``save_consolidated_best: false`` makes the run resume-only.

        DeepSpeed writes a sharded tag DIRECTORY, not a single file. Without the
        consolidated copy, ``discover_best_checkpoint``, campaign evaluation and
        ``spectramr infer`` all find nothing -- at the end of the run.
        """
        check_name = "deepspeed_consolidated_best_checkpoint"
        parallel = getattr(config, "parallel", None)
        strategy = getattr(parallel, "strategy", "none") if parallel else "none"
        if strategy != "deepspeed":
            return HealthCheckResult(
                True,
                check_name,
                f"strategy={strategy!r} is not deepspeed; n/a.",
                "info",
            )
        if parallel.deepspeed.save_consolidated_best:
            return HealthCheckResult(
                True,
                check_name,
                "A consolidated checkpoint_best will be written.",
                "info",
            )
        return HealthCheckResult(
            False,
            check_name,
            "parallel.deepspeed.save_consolidated_best=false: this run will be "
            "RESUME-ONLY. Inference and campaign evaluation read a single-file "
            "checkpoint and will find none.",
            "warning",
            category="deepspeed_consolidated_best_checkpoint",
            yaml_keys=["parallel.deepspeed.save_consolidated_best"],
            fix_hint="Leave save_consolidated_best: true unless the run is purely "
            "a resume experiment.",
        )

    def check_mamba_models_require_mamba_ssm(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Mamba-family models REQUIRE the official ``mamba_ssm`` kernel.

        A ``model_type`` containing ``mamba`` routes through
        :class:`~spectramr.models.blocks.mamba_block.MambaBlock`, which fails loud
        at construction without the selective-scan kernel — the Gated-Conv+GRU
        fallback is NOT an SSM (pitfall #9 / #16). This catches the missing
        dependency at audit time instead of the first train step, mirroring
        MambaBlock's runtime behaviour: absent kernel → error; absent kernel +
        ``SPECTRAMR_ALLOW_MAMBA_FALLBACK`` opt-in → warning (the run silently
        trains a non-SSM GRU).
        """
        from spectramr.models.blocks import mamba_block

        check_name = "mamba_models_require_mamba_ssm"
        model_type = str(getattr(getattr(config, "model", None), "model_type", "") or "")
        if "mamba" not in model_type.lower():
            return HealthCheckResult(
                True,
                check_name,
                f"model_type={model_type!r} is not a Mamba-family model.",
                "info",
            )
        if mamba_block._mamba_ssm_importable():
            return HealthCheckResult(
                True,
                check_name,
                f"mamba_ssm is importable for Mamba model_type={model_type!r}.",
                "info",
            )
        # `make install-mamba`, not the bare pip line: upstream downloads a
        # prebuilt wheel whose gencode list has no sm_70, so the import this check
        # tests would succeed while a V100 dies at the first kernel launch.
        install_hint = (
            "Install the official kernel: `make install-mamba` (needs CUDA + nvcc); "
            "verify with `python scripts/install_mamba_extra.py --verify-only`, which "
            "checks the built architectures and not just that `import mamba_ssm` works."
        )
        if mamba_block._mamba_fallback_allowed():
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"model_type={model_type!r} is a Mamba model but mamba_ssm is "
                    "not importable AND SPECTRAMR_ALLOW_MAMBA_FALLBACK is set: the run "
                    "will train a Gated-Conv+GRU approximation, NOT a true SSM — "
                    "results are not scientifically 'Mamba' (pitfall #16)."
                ),
                "warning",
                category="mamba_models_require_mamba_ssm",
                yaml_keys=["model.model_type"],
                fix_hint=install_hint + " Then unset SPECTRAMR_ALLOW_MAMBA_FALLBACK.",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                f"model_type={model_type!r} is a Mamba model but the official "
                "mamba_ssm selective-scan kernel is not importable — MambaBlock "
                "raises at construction (the GRU fallback is disabled by default)."
            ),
            "error",
            category="mamba_models_require_mamba_ssm",
            yaml_keys=["model.model_type"],
            fix_hint=install_hint,
        )

    # ──────────────────────────────────────────────────────────────────── #
    # F8 / E2 — CycleBloch requires a discriminator at audit-time.
    #
    # 2026-05-16 smoke audit found 7 runtime ``ValueError`` hits from
    # ``CycleBlochStrategy.train_step`` at
    # ``src/infrastructure/training/strategies/cycle_bloch_strategy.py:277``:
    # ``"CycleBlochStrategy requires a discriminator. Either (a) add a
    # ``discriminator:`` section to the experiment YAML under ``model:``...".
    # The check is on ``self.env.discriminator`` which is built by the
    # ModelBuilder from ``config.model.discriminator_component``. When the
    # YAML omits that block, discriminator is None and the strategy bails
    # at the first training step. Hoisting to audit-time per CLAUDE.md
    # pitfall #10. Audit anchor: TODO/audit/smoke_audit_20260516.md §F8.
    # ──────────────────────────────────────────────────────────────────── #

    _CYCLE_BLOCH_TRAINING_MODES = frozenset({"cycle_bloch"})

    def check_cycle_bloch_requires_discriminator(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """``CycleBlochStrategy`` requires ``model.discriminator_component``.

        The strategy unconditionally accesses ``self.env.discriminator`` in
        its training step, which the ``ModelBuilder`` only populates when
        ``config.model.discriminator_component`` is non-None.
        """
        check_name = "cycle_bloch_requires_discriminator"
        training = getattr(config, "training", None)
        if training is None:
            return HealthCheckResult(
                True,
                check_name,
                "no training block",
                "info",
            )
        mode = (getattr(training, "training_mode", "") or "").lower()
        if mode not in self._CYCLE_BLOCH_TRAINING_MODES:
            return HealthCheckResult(
                True,
                check_name,
                f"training_mode={mode!r} is not cycle_bloch.",
                "info",
            )
        model = getattr(config, "model", None)
        disc = getattr(model, "discriminator_component", None) if model is not None else None
        if disc is None:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"training.training_mode={mode!r} requires "
                    "config.model.discriminator_component to be populated. "
                    "CycleBlochStrategy raises ValueError at the first "
                    "training step when self.env.discriminator is None."
                ),
                "error",
                category="strategy_requires_schema",
                yaml_keys=["training.training_mode", "model.discriminator_component"],
                fix_hint=(
                    "Add a `model.discriminator_component:` block to the "
                    "YAML. Minimal form: "
                    "`{discriminator_type: patch_gan_2d, in_channels: 1}`. "
                    "See src/spectramr/config/schemas/templates/v1.0_reference.yaml "
                    "for the canonical layout."
                ),
            )
        return HealthCheckResult(
            True,
            check_name,
            "cycle_bloch: discriminator_component populated.",
            "info",
        )

    # ─────────────────────────────────────────────────────────────── #
    # F7 / 2026-05-17 round 4 — audit-assumption transparency
    #
    # ``check_domain_alignment`` (Tier-1) statically derives the
    # expected per-sample channel count from
    # ``data.coil_processing_mode + dataset_type`` and compares it to
    # ``model.in_channels``. Three known blind spots let configs slip
    # past the audit and fail at runtime with ``[DomainMismatch]``
    # (63 hits in 2026-05-16 smoke):
    #
    #   1. Models in ``_INPUT_CONCAT_MODELS`` bypass the strict
    #      equality check (runtime concat changes the input shape).
    #      Today: silent info. After F7: warning, so ``--strict``
    #      flags the bypass.
    #   2. ``adapters.pre_model`` chains run BETWEEN the
    #      coil-processing op and the model, so the static channel
    #      derivation may be wrong even when the YAML "looks right".
    #   3. ``coil_processing_mode='none'`` defers the channel count
    #      to the per-file h5 header — the audit cannot resolve it.
    #
    # Audit anchor: TODO/audit/smoke_audit_20260516.md §F7.
    # ─────────────────────────────────────────────────────────────── #

    def check_channel_audit_assumptions(
        self,
        config: TrainingSettings,
    ) -> list[HealthCheckResult]:
        """Make channel-audit blind spots visible to ``--strict`` mode.

        Returns one ``warning``-severity result per blind spot
        encountered, plus an ``info`` result when the static
        derivation is sufficient (no blind spot triggered). The smoke
        wrapper runs ``--strict``, so each warning hard-gates the
        train invocation, surfacing configs that would otherwise pass
        Tier 0+1 and crash at the first training batch.
        """
        check_name = "channel_audit_assumptions"
        results: list[HealthCheckResult] = []

        model = getattr(config, "model", None)
        data = getattr(config, "data", None)
        adapters = getattr(config, "adapters", None)

        model_type = (getattr(model, "model_type", "") or "") if model is not None else ""
        if self._expects_input_concat(model):
            in_ch = getattr(model, "in_channels", None)
            # F6 / 2026-05-20 — input-concat is *by design* (the strategy
            # advertises it in _INPUT_CONCAT_MODELS). When the user has
            # explicitly set ``model.in_channels``, the static check has
            # nothing further to verify; surfacing this as a strict-mode
            # warning produced 70 spurious failures in the 2026-05-19
            # smoke run. Demote to ``info`` so --probe / runtime DC check
            # remain the source of truth.
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name=check_name,
                    message=(
                        f"model_type={model_type!r} is in "
                        f"_INPUT_CONCAT_MODELS and this arm's model_kwargs "
                        f"resolve to an actual concat (#1387) — the strategy "
                        f"concatenates extra conditioning channels onto "
                        f"the input at runtime, so check_domain_alignment "
                        f"bypassed strict equality. "
                        f"model.in_channels={in_ch} is the declared "
                        f"PRE-concat width; the backbone is built wider. The "
                        f"runtime [DomainMismatch] check at "
                        f"strategies/base.py:660 remains the source of truth."
                    ),
                    severity="info",
                    category="audit_assumption_unverified",
                    yaml_keys=["model.model_type", "model.in_channels"],
                    fix_hint=(
                        "Run `python -m spectramr.cli audit --probe` to verify "
                        "the post-concat shape via a synthetic forward "
                        "pass when changing model_kwargs."
                    ),
                )
            )

        pre_model_chain = getattr(adapters, "pre_model", None) if adapters is not None else None
        if pre_model_chain:
            adapter_names = [getattr(step, "name", None) for step in pre_model_chain]
            adapter_names = [n for n in adapter_names if n]
            # F6b / 2026-05-20 — when every adapter in the chain has a
            # deterministic, audit-known channel transform (entries in
            # ``_ADAPTER_CHANNEL_EFFECTS``), the static channel check
            # ``check_adapter_chain_channel_resolution`` already verifies
            # the resolved width — no blind spot remains. Demote to
            # ``info``; reserve the strict-mode warning for chains that
            # contain at least one unknown-effect adapter.
            all_known = bool(adapter_names) and all(
                n in self._ADAPTER_CHANNEL_EFFECTS for n in adapter_names
            )
            if all_known:
                results.append(
                    HealthCheckResult(
                        passed=True,
                        check_name=check_name,
                        message=(
                            f"adapters.pre_model chain "
                            f"({','.join(adapter_names)}) is entirely "
                            f"composed of audit-known channel-transform "
                            f"adapters; check_adapter_chain_channel_resolution "
                            f"covers the post-adapter width statically."
                        ),
                        severity="info",
                        category="audit_assumption_resolved",
                        yaml_keys=["adapters.pre_model", "model.in_channels"],
                        fix_hint=None,
                    )
                )
            else:
                unknown = [n for n in adapter_names if n not in self._ADAPTER_CHANNEL_EFFECTS]
                results.append(
                    HealthCheckResult(
                        passed=False,
                        check_name=check_name,
                        message=(
                            f"adapters.pre_model declared "
                            f"({len(pre_model_chain)} step(s): "
                            f"{','.join(adapter_names) or '?'}); "
                            f"{len(unknown)} adapter(s) lack a registered "
                            f"channel-effect entry ({','.join(unknown)}). "
                            f"The static check_domain_alignment derivation "
                            f"may not reflect the actual input shape the "
                            f"model receives. Verify via --probe or by "
                            f"inspecting first-batch shape in the train log."
                        ),
                        severity="warning",
                        category="audit_assumption_unverified",
                        yaml_keys=["adapters.pre_model", "model.in_channels"],
                        fix_hint=(
                            "If the pre_model adapters are channel-preserving "
                            "(e.g. identity), annotate that in a YAML comment "
                            "next to the chain. Otherwise rerun audit with "
                            "--probe to confirm the post-adapter shape "
                            "matches model.in_channels."
                        ),
                    )
                )

        coil_mode = (
            getattr(getattr(data, "coils", None), "processing_mode", None)
            if data is not None
            else None
        )
        if coil_mode == "none":
            in_ch = getattr(model, "in_channels", None) if model is not None else None
            target_ch = data.domain.target_channels if data is not None else None
            # F6c / 2026-05-20 — when the user has explicitly declared
            # ``data.domain.target_channels``, they've stated the expected
            # per-sample channel count even though the file header is
            # the runtime source of truth. Demote to ``info`` IF
            # ``in_channels`` matches ``target_channels``; the runtime
            # DomainMismatch check still guards the actual header value.
            if in_ch is not None and target_ch is not None and int(in_ch) == int(target_ch):
                results.append(
                    HealthCheckResult(
                        passed=True,
                        check_name=check_name,
                        message=(
                            f"coil_processing_mode='none'; "
                            f"model.in_channels={in_ch} matches "
                            f"data.domain.target_channels={target_ch}, so the "
                            f"static channel count is consistent with the "
                            f"user-declared expectation. Runtime "
                            f"[DomainMismatch] check remains the source "
                            f"of truth for the actual h5 header."
                        ),
                        severity="info",
                        category="audit_assumption_resolved",
                        yaml_keys=[
                            "data.coil_processing_mode",
                            "data.domain.target_channels",
                            "model.in_channels",
                        ],
                        fix_hint=None,
                    )
                )
            else:
                # F32 / 2026-05-22 — this is an *unverifiable assumption*, not a
                # defect: coil_processing_mode='none' deliberately defers the
                # per-sample channel count to the h5 header, and for asymmetric
                # recon (e.g. 8-coil k-space input → 1-channel magnitude target)
                # there is NO static way to confirm it — in_channels (input
                # coils) legitimately differs from target_channels (output
                # magnitude). Emitting it at ``warning`` made ``--strict`` exit
                # 2 and hard-gate 33 intentional multi-coil arms in smoke
                # 20260521 (every exp_11* / baseline_* / conformal arm). The
                # runtime [DomainMismatch] check at strategies/base.py:660 is
                # the real source of truth, so this is informational: keep the
                # guidance message but don't fail the audit.
                results.append(
                    HealthCheckResult(
                        passed=True,
                        check_name=check_name,
                        message=(
                            f"coil_processing_mode='none' defers the channel "
                            f"count to the per-file h5 header — the audit "
                            f"cannot resolve it statically (informational). "
                            f"model.in_channels={in_ch} must match the "
                            f"dataset's per-sample channel count at runtime; "
                            f"failure surfaces as [DomainMismatch] from "
                            f"strategies/base.py:660."
                        ),
                        severity="info",
                        category="audit_assumption_unverified",
                        yaml_keys=[
                            "data.coil_processing_mode",
                            "model.in_channels",
                        ],
                        fix_hint=(
                            "Set an explicit coil_processing_mode (rss, "
                            "magnitude, svd, flatten) so the audit can "
                            "verify channel alignment statically, set "
                            "data.domain.target_channels=<expected> to confirm the "
                            "intended count, or run `--probe` to check at "
                            "audit time."
                        ),
                    )
                )

        if not results:
            results.append(
                HealthCheckResult(
                    True,
                    check_name,
                    ("No channel-audit blind spots detected (static derivation suffices)."),
                    "info",
                )
            )

        return results

    # ─────────────────────────────────────────────────────────────── #
    # F2 / 2026-05-17 round 6 — visualization-emission audit
    #
    # 147 experiments PASSED their smoke run but emitted ZERO validation
    # images in 2026-05-16 (E2/F2 in the audit doc). Root-cause analysis
    # in rounds 1-2 identified at least three distinct sub-causes; the
    # cleanest audit-time gate flags the YAML-level pattern:
    #
    #     logging.enable_viz: true
    #     logging.save_validation_images: true     # (or unset → default true)
    #     validation.visualization_interval > training.max_iterations
    #
    # When the interval exceeds max_iterations, the val loop never
    # reaches a "save" tick — the toggle is on but no images ever land
    # on disk. The smoke wrapper overrides ``visualization_interval=
    # ${TRAIN_ITERS}`` at runtime, so the audit-time YAML view masks the
    # bug for smoke runs but a real training launch is silently broken.
    #
    # Audit anchor: TODO/audit/smoke_audit_20260516.md §F2.
    # ─────────────────────────────────────────────────────────────── #

    def check_visualization_interval_reachable(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """``validation.visualization_interval`` must be ≤ ``training.max_iterations``.

        Otherwise the training loop terminates before any visualization
        tick fires and validation PNGs are never saved, even with
        ``enable_viz: true``. This is the silent regression that
        produced 147 experiments with PASS + zero PNGs in the
        2026-05-16 smoke run.
        """
        check_name = "visualization_interval_reachable"
        logging_cfg = getattr(config, "logging", None)
        validation = getattr(config, "validation", None)
        training = getattr(config, "training", None)
        if logging_cfg is None or validation is None or training is None:
            return HealthCheckResult(
                True,
                check_name,
                "missing one of logging/validation/training blocks",
                "info",
            )
        enable_viz = getattr(logging_cfg, "enable_viz", None)
        save_val_imgs = logging_cfg.images.save_validation if logging_cfg else None
        # save_validation_images defaults to True at schema layer
        # (src/config/schemas/logging.py:169-172). Treat None as True.
        if save_val_imgs is None:
            save_val_imgs = True
        if not (enable_viz and save_val_imgs):
            return HealthCheckResult(
                True,
                check_name,
                (
                    f"viz disabled: enable_viz={enable_viz}, "
                    f"save_validation_images={save_val_imgs} — "
                    f"interval-reachability check moot."
                ),
                "info",
            )
        viz_interval = validation.visualization.interval if validation else None
        max_iter = getattr(training, "max_iterations", None)
        if viz_interval is None or max_iter is None:
            return HealthCheckResult(
                True,
                check_name,
                "visualization_interval or max_iterations unset — defer",
                "info",
            )
        try:
            viz_i = int(viz_interval)
            max_i = int(max_iter)
        except (TypeError, ValueError):
            return HealthCheckResult(
                True,
                check_name,
                "non-numeric viz interval / max_iterations — defer",
                "info",
            )
        if viz_i > max_i:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"validation.visualization_interval={viz_i} > "
                    f"training.max_iterations={max_i}. The training "
                    f"loop will terminate before any viz tick fires, "
                    f"and validation PNGs will NEVER be saved despite "
                    f"enable_viz=true. (This is the silent regression "
                    f"that produced 147 PASS-with-no-images "
                    f"experiments in the 2026-05-16 smoke run.) The "
                    f"smoke wrapper overrides "
                    f"visualization_interval=TRAIN_ITERS at runtime, "
                    f"so the regression is masked under smoke but real "
                    f"training is silently broken."
                ),
                "warning",
                category="viz_interval_unreachable",
                yaml_keys=[
                    "validation.visualization_interval",
                    "training.max_iterations",
                ],
                fix_hint=(
                    "Set validation.visualization_interval to a value "
                    "≤ training.max_iterations (typically max_iter/10 "
                    "or max_iter/20 for ~10-20 validation snapshots)."
                ),
            )
        return HealthCheckResult(
            True,
            check_name,
            (
                f"validation.visualization_interval={viz_i} ≤ "
                f"training.max_iterations={max_i} — viz ticks will fire."
            ),
            "info",
        )

    # ─────────────────────────────────────────────────────────────── #
    # F5 / 2026-05-17 round 6 — hardcoded cluster-path detection
    #
    # 68+13 hits in 2026-05-16 smoke (E16) came from YAMLs hardcoding
    # ``/project/<group>/<user>/spectramr/data/manifests/...`` as the
    # manifest path. On the local box those paths don't exist; on the
    # cluster they only exist after a manual sync. Per CLAUDE.md
    # pitfall #16, the fix is to use ``PathResolver.resolve()`` so the
    # path adapts to the host. The audit-time check catches the
    # hardcoded form and refuses to proceed.
    #
    # F5b / 2026-05-18 — the May 18 cluster smoke surfaced 2 arms that
    # passed audit but failed training start-up with the SAME check.
    # Root cause: a YAML with ``data.data_root: ./data`` (or schema
    # default) was joined against ``PROJECT_ROOT=/project/<user>/...``
    # by ``PathNormalizer`` after the audit had inspected the raw YAML
    # value but before training re-ran the check on the resolved
    # value. The fix exempts paths that live *under* the user's own
    # configured ``SPECTRAMR_DATA_ROOT`` / ``PROJECT_ROOT`` — that's where
    # the data is supposed to be, not a leaked literal.
    #
    # Audit anchor: TODO/audit/smoke_audit_20260516.md §F5,
    # tests_experiments/smoke_test/smoke_test_all_20260518_120147.log
    # (cs_mno_burgers, experiment_75_xdiffusion_latent_bridge).
    # ─────────────────────────────────────────────────────────────── #

    #: Shared-cluster mount roots a config must never hardcode.
    #:
    #: These are GENERIC prefixes, not one site's allocation names. The list
    #: used to enumerate three specific allocations, which made the check fire
    #: only for the sites that happened to be listed -- every other cluster's
    #: hardcoded path sailed through, and the enumeration additionally shipped
    #: this site's identity in the package source. Any absolute path under a
    #: shared mount is unportable regardless of whose allocation it names, so
    #: the general prefix is both identity-free and a strictly stronger check.
    #: The docstring of :meth:`check_hardcoded_cluster_paths` already promised
    #: exactly this behaviour.
    #:
    #: A path under the user's OWN configured root stays exempt --
    #: see :meth:`_user_configured_roots`.
    _FORBIDDEN_PATH_PREFIXES: frozenset[str] = frozenset(
        {
            "/project/",
            "/scratch/",
        }
    )

    _PATH_FIELDS_TO_CHECK = (
        (
            "data.data_root",
            lambda c: getattr(getattr(c, "data", None), "source", None) and c.data.source.root,
        ),
        (
            "data.index_path",
            lambda c: (
                getattr(getattr(c, "data", None), "source", None) and c.data.source.index_path
            ),
        ),
        (
            "data.validation_index_path",
            lambda c: (
                getattr(getattr(c, "data", None), "source", None)
                and c.data.source.validation_index_path
            ),
        ),
        (
            "data.test_index_path",
            lambda c: getattr(getattr(c, "data", None), "test_index_path", None),
        ),
    )

    @staticmethod
    def _user_configured_roots() -> tuple[str, ...]:
        """Return the user's configured project/data roots, normalised with a
        trailing slash so prefix checks don't false-positive on a partial
        directory name.

        A path that lives *under* the user's own ``PROJECT_ROOT`` /
        ``SPECTRAMR_DATA_ROOT`` / ``SPECTRAMR_ROOT`` is not a "leak" — it is
        exactly where the data is supposed to live. We want this check to
        fire only when a YAML hardcodes some *other* team's cluster prefix
        (a literal copied from an old config or an upstream sibling repo).

        F5c / 2026-05-20 — Auto-detect the running user's own cluster mount
        from ``$USER`` + ``cwd`` so the check exempts the legitimate case
        ("I am ``$USER`` running from ``/project/<group>/$USER/...``") without
        requiring the cluster ``.env`` to set ``SPECTRAMR_DATA_ROOT``
        explicitly. The auto-detected root is the shortest subtree under a
        forbidden prefix that ENDS at the running user's own segment — a
        colleague's leaked literal (``/project/<group>/<someone-else>/``)
        never matches and still fires.

        ``$USER`` may sit at any depth under the mount: ``/scratch/$USER/``
        and ``/project/<group>/$USER/`` are both real layouts, so the segment
        is searched for rather than assumed to be the first one.
        """
        import os
        from pathlib import Path

        roots: list[str] = []
        for env_var in ("SPECTRAMR_DATA_ROOT", "PROJECT_ROOT", "SPECTRAMR_ROOT"):
            v = os.environ.get(env_var)
            if not v:
                continue
            v = v.rstrip("/") + "/"
            if v not in roots:
                roots.append(v)

        # Auto-detect: if cwd lives under a forbidden cluster prefix AND
        # the running user's name appears as a path segment under that
        # prefix, treat the user's account-level subtree as a configured
        # root. This is the "I AM the cluster owner; this IS my mount"
        # case that the env-var exemption was meant to cover.
        user = os.environ.get("USER", "")
        if user:
            try:
                cwd_parts = Path.cwd().resolve().parts
            except (OSError, RuntimeError):
                cwd_parts = ()
            for prefix in ConfigHealthChecker._FORBIDDEN_PATH_PREFIXES:
                prefix_parts = Path(prefix).parts
                # ``cwd`` starts with the forbidden prefix?
                if len(cwd_parts) < len(prefix_parts) + 1:
                    continue
                if cwd_parts[: len(prefix_parts)] != prefix_parts:
                    continue
                # SOME segment under the mount must equal $USER -- not
                # necessarily the first, since ``/project/<group>/$USER/`` is
                # as common a layout as ``/scratch/$USER/``. Stop at the
                # shallowest match so the exempted subtree is the user's own
                # directory and no wider; a colleague's directory under the
                # same mount never matches, so their leaked literal still
                # fires.
                for depth in range(len(prefix_parts), len(cwd_parts)):
                    if cwd_parts[depth] != user:
                        continue
                    auto_root = str(Path(*cwd_parts[: depth + 1])) + "/"
                    if auto_root not in roots:
                        roots.append(auto_root)
                    break
        return tuple(roots)

    def check_hardcoded_cluster_paths(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Reject hardcoded cluster paths under ``/project/`` or ``/scratch/``.

        A path that resolves *under* the user's own configured
        ``SPECTRAMR_DATA_ROOT`` or ``PROJECT_ROOT`` is exempt: it indicates
        the user has correctly pointed the framework at their cluster
        mount, not that they hardcoded somebody else's prefix.
        """
        check_name = "hardcoded_cluster_paths"
        user_roots = self._user_configured_roots()
        offenders = []
        for field_name, getter in self._PATH_FIELDS_TO_CHECK:
            value = getter(config)
            if value is None:
                continue
            value_str = str(value)
            # Skip if the path lives under the user's own configured root.
            if any(value_str.startswith(r) for r in user_roots):
                continue
            for forbidden in self._FORBIDDEN_PATH_PREFIXES:
                if value_str.startswith(forbidden):
                    offenders.append((field_name, value_str, forbidden))
                    break
        if not offenders:
            return HealthCheckResult(
                True,
                check_name,
                "no hardcoded cluster paths detected",
                "info",
            )
        first_field, first_val, first_prefix = offenders[0]
        return HealthCheckResult(
            False,
            check_name,
            (
                f"{first_field}={first_val!r} starts with hardcoded "
                f"cluster prefix {first_prefix!r}. Per CLAUDE.md "
                f"pitfall #16, use PathResolver.resolve() or a "
                f"relative path under data/manifests/. "
                f"({len(offenders)} field(s) affected.) "
                f"If this IS your cluster mount, set "
                f"SPECTRAMR_DATA_ROOT={first_prefix.rstrip('/')}/<your-account>/ "
                f"and the check will exempt paths under that root."
            ),
            "error",
            category="hardcoded_cluster_path",
            yaml_keys=[f for f, _, _ in offenders],
            fix_hint=(
                "Replace the hardcoded path with a relative path "
                "(e.g. 'data/manifests/foo.json') and rely on "
                "PathResolver to resolve it via DATA_ROOT or symlinks. "
                "Both local and cluster setups then work without YAML "
                "edits. Alternatively, set SPECTRAMR_DATA_ROOT to your "
                "own cluster mount so the check exempts paths under it."
            ),
        )

    # ─────────────────────────────────────────────────────────────── #
    # F7-Hoist / 2026-05-17 round 6 — adapter-aware channel resolution
    #
    # F7 (round 4) added a transparency warning whenever
    # ``adapters.pre_model`` was declared, because the static
    # ``check_domain_alignment`` couldn't model the adapter's effect.
    # F7-Hoist promotes the warning to a hard error for the FOUR
    # known adapters whose channel transform is deterministic:
    #
    #   adapter                                effect (real-valued in)
    #   -----------------------------------    ----------------------
    #   rss_coils_to_magnitude                 (..., C, ...) → (..., 1, ...)
    #   magnitude_from_complex                 (..., 2C, ...) → (..., C, ...)
    #   real_imag_interleave_to_complex        (..., 2C, ...) → (..., C, ...) complex
    #   complex_to_real_imag_interleave        (..., C, ...) complex → (..., 2C, ...) real
    #
    # See ``src/data/adapters/channels.py`` for the implementations.
    # When all chain steps are known, the audit folds the expected
    # output channel count and compares it to ``model.in_channels``.
    # Unknown adapter names fall back to the round-4 warning.
    #
    # Audit anchor: TODO/audit/smoke_audit_20260516.md §F7-Hoist.
    # ─────────────────────────────────────────────────────────────── #

    # adapter_name → callable(input_channels: int) → output_channels: int
    _ADAPTER_CHANNEL_EFFECTS = {
        "rss_coils_to_magnitude": (lambda c: 1),
        # magnitude_from_complex: 2C interleaved → C; treat odd C as
        # identity (already magnitude). The most common smoke YAML
        # usage is on already-1ch data so the identity branch fires.
        "magnitude_from_complex": (lambda c: max(c // 2, 1) if c % 2 == 0 else c),
        "real_imag_interleave_to_complex": (lambda c: max(c // 2, 1) if c % 2 == 0 else c),
        "complex_to_real_imag_interleave": (lambda c: c * 2),
        # F-FFT-ADAPTER-CHANNEL-EFFECT / 2026-05-20 — the FFT pair is
        # channel-preserving (the spatial transform doesn't change C).
        # Round 8j's FNO YAML wires ``pre_model: [fft_image_to_kspace,
        # complex_to_real_imag_interleave]`` to bridge a 1-ch image
        # into a 2-ch real-as-interleaved-complex k-space — without
        # this entry the channel-audit warning fires at strict-mode
        # for the FFT step despite its effect being deterministic.
        "fft_image_to_kspace": (lambda c: c),
        "ifft_kspace_to_image": (lambda c: c),
        # The 3-D ↔ 2-D rank adapters preserve channel count.
        "slice_3d_to_2d": (lambda c: c),
        "gather_2d_to_3d": (lambda c: c),
        # F-UNFLATTEN-1D pair (added round 8h) is also channel-preserving.
        "flatten_spatial_to_1d": (lambda c: c),
        "unflatten_1d_to_spatial": (lambda c: c),
    }

    def check_adapter_chain_channel_resolution(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Apply known ``adapters.pre_model`` effects, then compare to ``in_channels``.

        Returns one of:

        - ``info`` (passed): no pre_model chain, or chain ends at a
          shape that matches ``model.in_channels``.
        - ``error`` (passed=False, severity="error"): chain is fully
          known but ends at a shape that DIFFERS from
          ``model.in_channels``. This is the runtime [DomainMismatch]
          regression that escapes the round-4 F7 warning.
        - ``info`` with note: chain contains at least one unknown
          adapter name — the round-4 F7 transparency warning
          (``check_channel_audit_assumptions``) carries that case.
        """
        check_name = "adapter_chain_channel_resolution"
        adapters = getattr(config, "adapters", None)
        pre_model_chain = getattr(adapters, "pre_model", None) if adapters is not None else None
        if not pre_model_chain:
            return HealthCheckResult(
                True,
                check_name,
                "no adapters.pre_model chain declared",
                "info",
            )

        model = getattr(config, "model", None)
        data = getattr(config, "data", None)
        if model is None or data is None:
            return HealthCheckResult(
                True,
                check_name,
                "no model/data block — chain effect not derivable",
                "info",
            )

        in_channels = getattr(model, "in_channels", None)
        if in_channels is None:
            return HealthCheckResult(
                True,
                check_name,
                "model.in_channels unset — chain effect not derivable",
                "info",
            )

        # Models whose strategy concatenates an extra reference / conditioning
        # tensor (S-maps, HF reference, paired contrast) at runtime legitimately
        # declare an in_channels LARGER than the adapter-chain output — the concat
        # is invisible to the static chain walk. Skip them here, mirroring the
        # domain_alignment in_channels exemption for the same model sets.
        model_type = getattr(model, "model_type", None)
        if self._expects_input_concat(model) or self._is_paired_contrast(config, model_type):
            return HealthCheckResult(
                True,
                check_name,
                (
                    f"model_type={model_type!r} concatenates a runtime "
                    f"conditioning tensor onto the input — static pre_model "
                    f"channel resolution skipped (matches the domain_alignment "
                    f"in_channels exemption)."
                ),
                "info",
            )

        # Derive the chain's input channel count from coil_processing_mode +
        # dataset_type. Reuse the existing helper for SSOT.
        coil_mode = getattr(getattr(data, "coils", None), "processing_mode", None)
        dataset_type = getattr(data, "dataset_type", None)
        num_virtual_coils = getattr(getattr(data, "coils", None), "num_virtual_coils", None)
        expected_input, reason = self._derive_expected_channels(
            coil_mode=coil_mode,
            dataset_type=dataset_type,
            num_virtual_coils=num_virtual_coils,
            in_channels=in_channels,
        )
        if expected_input is None:
            return HealthCheckResult(
                True,
                check_name,
                (
                    f"pre_model chain present but pre-chain channel "
                    f"count cannot be derived statically ({reason}). "
                    f"Static derivation deferred — see round-4 F7 "
                    f"warning."
                ),
                "info",
            )

        # Walk the chain. If ANY adapter name is unknown, fall back to
        # round-4 F7 transparency (already emitted as a warning by
        # ``check_channel_audit_assumptions``).
        current_channels = expected_input
        adapter_trail = []
        for step in pre_model_chain:
            adapter_name = getattr(step, "name", None)
            if adapter_name is None or adapter_name not in self._ADAPTER_CHANNEL_EFFECTS:
                return HealthCheckResult(
                    True,
                    check_name,
                    (
                        f"pre_model chain contains adapter "
                        f"{adapter_name!r} not in the F7-Hoist "
                        f"known-effect set "
                        f"{sorted(self._ADAPTER_CHANNEL_EFFECTS)}. "
                        f"Static channel resolution deferred — see "
                        f"round-4 F7 warning."
                    ),
                    "info",
                )
            effect = self._ADAPTER_CHANNEL_EFFECTS[adapter_name]
            new_channels = effect(current_channels)
            adapter_trail.append(f"{adapter_name}: {current_channels} → {new_channels}")
            current_channels = new_channels

        # All adapters are known — compare final shape to model.in_channels.
        if current_channels == in_channels:
            return HealthCheckResult(
                True,
                check_name,
                (
                    f"adapter chain resolves cleanly: "
                    f"data={expected_input}ch → "
                    f"{' → '.join(adapter_trail)} matches "
                    f"model.in_channels={in_channels}."
                ),
                "info",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                f"adapter chain mismatch: data starts at "
                f"{expected_input}ch ({reason}); after "
                f"{' → '.join(adapter_trail)} the model receives "
                f"{current_channels}ch, but model.in_channels="
                f"{in_channels}. This will raise [DomainMismatch] at "
                f"the first training step."
            ),
            "error",
            category="adapter_chain_channel_mismatch",
            yaml_keys=[
                "adapters.pre_model",
                "model.in_channels",
                "data.coil_processing_mode",
            ],
            fix_hint=(
                f"Either change model.in_channels to "
                f"{current_channels}, or rework the adapters.pre_model "
                f"chain so the final shape matches "
                f"model.in_channels={in_channels}. Reference: "
                f"src/data/adapters/channels.py for adapter semantics."
            ),
        )

    # ─────────────────────────────────────────────────────────────── #
    # F-OUT / 2026-05-17 round 6 — training.output_dir convention
    #
    # The canonical results directory is
    # ``experiments/results/<experiment_name>``. 197 YAMLs in the
    # 2026-05-16 smoke run either omitted ``training.output_dir``
    # entirely (77), used the deprecated ``linear_configs/`` legacy
    # subdirectory (85), or pointed at the wrong prefix (35:
    # ``experiments/active/<name>`` self-referential,
    # ``experiments/outputs/``, ``experiments/test_experiments/``,
    # etc.). The runtime effect: validation PNGs, mosaics, and
    # checkpoints land in arbitrary locations, breaking
    # ``mosaic_validation.py`` discovery and the
    # ``no_validation_images_report``.
    #
    # Audit anchor: TODO/audit/smoke_audit_20260516.md §F-OUT.
    # ─────────────────────────────────────────────────────────────── #

    _OUTPUT_DIR_DEPRECATED_PREFIXES: frozenset[str] = frozenset(
        {
            "experiments/results/linear_configs/",
            "experiments/active/",
            "experiments/outputs/",
            "experiments/test_experiments/",
        }
    )

    def check_best_metric_matches_early_stopping(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """``metrics.best_metric_name`` must not contradict the live selector.

        Checkpoint selection has ONE live owner: ``early_stopping.metric``,
        resolved by ``EarlyStoppingService`` and consumed by ``save_best``.
        ``metrics.best_metric_name`` is inert for selection -- it is read by the
        witness check and the reporting layer, nothing else -- so an arm whose
        two spellings disagree writes a report naming a metric that did NOT pick
        its checkpoint.

        Measured across ``experiments/``: 126 arms disagree, and on **75** of
        them early stopping is enabled, so a wrong selection really happens.

        Severities:

        - ``info`` (passes): the two agree, or only one is declared, or early
          stopping is disabled so no selection happens and the key is decorative.
        - ``error``: they disagree while early stopping is enabled.
        """
        check_name = "best_metric_matches_early_stopping"
        early = getattr(config, "early_stopping", None)
        metrics = getattr(config, "metrics", None)
        if early is None or metrics is None:
            return HealthCheckResult(True, check_name, "no selector blocks declared", "info")
        if not getattr(early, "enabled", False):
            return HealthCheckResult(
                True,
                check_name,
                "early stopping disabled — no checkpoint selection happens, so "
                "metrics.best_metric_name selects nothing either way",
                "info",
            )
        # DECLARED, not defaulted. `metrics.best_metric_name` defaults to
        # 'val_loss' and `best_metric_mode` to MIN, so an arm that sets only
        # `early_stopping.metric` parses with a value it never wrote -- and
        # comparing that would report a contradiction the author did not make,
        # on 109 inprogress arms. Pydantic records what was actually supplied.
        supplied = getattr(metrics, "model_fields_set", None)
        if supplied is not None and "best_metric_name" not in supplied:
            return HealthCheckResult(
                True,
                check_name,
                "metrics.best_metric_name is not declared — nothing contradicts "
                "the live early_stopping selector",
                "info",
            )
        live_name = getattr(early, "metric", None)
        declared_name = getattr(metrics, "best_metric_name", None)
        # `.value` first: both spellings resolve to a `MetricMode`, and printing
        # the enum repr would put "MetricMode.MAX" in a message whose whole job
        # is to be compared against a YAML line that reads `max`.
        def _mode(raw: object) -> str:
            return str(getattr(raw, "value", raw) or "")

        live_mode = _mode(getattr(early, "mode", ""))
        declared_mode = (
            _mode(getattr(metrics, "best_metric_mode", ""))
            if supplied is None or "best_metric_mode" in supplied
            else ""
        )
        clashes = []
        if live_name and declared_name and str(live_name) != str(declared_name):
            clashes.append(
                f"metrics.best_metric_name={declared_name!r} vs "
                f"early_stopping.metric={live_name!r}"
            )
        if live_mode and declared_mode and live_mode != declared_mode:
            clashes.append(
                f"metrics.best_metric_mode={declared_mode!r} vs "
                f"early_stopping.mode={live_mode!r}"
            )
        if not clashes:
            return HealthCheckResult(
                True,
                check_name,
                f"selector agrees across both spellings ({live_name!r}, {live_mode!r})",
                "info",
            )
        return HealthCheckResult(
            False,
            check_name,
            "checkpoint selection is declared two ways and they disagree: "
            + "; ".join(clashes)
            + ". early_stopping is the LIVE one; metrics.best_metric_name is read "
            "only by the witness check and the report, so this run would select on "
            f"{live_name!r} while its report named {declared_name!r}. Make them "
            "match, or drop metrics.best_metric_name.",
            "error",
        )

    def check_output_dir_convention(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """``training.output_dir`` must be ``experiments/results/<name>``.

        Severities:

        - ``info`` (passes): output_dir starts with
          ``experiments/results/`` and isn't under
          ``linear_configs/``.
        - ``warning`` (passed=False, severity="warning"):
          output_dir absent → defaults to a generic location;
          mosaic discovery breaks.
        - ``error`` (passed=False, severity="error"): output_dir
          under one of the deprecated prefixes.
        """
        check_name = "output_dir_convention"
        training = getattr(config, "training", None)
        if training is None:
            return HealthCheckResult(
                True,
                check_name,
                "no training block",
                "info",
            )
        output_dir = getattr(training, "output_dir", None)
        if not output_dir:
            return HealthCheckResult(
                False,
                check_name,
                (
                    "training.output_dir is absent — validation PNGs, "
                    "mosaics, and checkpoints land in the runtime "
                    "default which breaks mosaic_validation.py "
                    "discovery and the no_validation_images_report. "
                    "Set training.output_dir: "
                    "experiments/results/<experiment_name>."
                ),
                "warning",
                category="output_dir_missing",
                yaml_keys=["training.output_dir"],
                fix_hint=(
                    "Add `training.output_dir: "
                    "experiments/results/<your_experiment_name>` to "
                    "the YAML. The smoke wrapper + mosaic tooling "
                    "auto-discover under this prefix."
                ),
            )
        path = str(output_dir)
        for bad in self._OUTPUT_DIR_DEPRECATED_PREFIXES:
            if path.startswith(bad):
                return HealthCheckResult(
                    False,
                    check_name,
                    (
                        f"training.output_dir='{path}' uses the "
                        f"deprecated prefix '{bad}'. The canonical "
                        f"location is 'experiments/results/"
                        f"<experiment_name>'. Outputs under the "
                        f"deprecated prefix are not picked up by "
                        f"mosaic_validation.py or process_smoke_log.py."
                    ),
                    "error",
                    category="output_dir_deprecated",
                    yaml_keys=["training.output_dir"],
                    fix_hint=(
                        "Rewrite training.output_dir to "
                        "'experiments/results/<experiment_name>' "
                        "(use the YAML stem as the name)."
                    ),
                )
        if not path.startswith("experiments/results/"):
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"training.output_dir='{path}' does not start "
                    f"with 'experiments/results/'. The smoke tooling "
                    f"expects this convention; non-standard paths "
                    f"break mosaic_validation.py auto-discovery."
                ),
                "warning",
                category="output_dir_nonstandard",
                yaml_keys=["training.output_dir"],
                fix_hint=(
                    "Use 'experiments/results/<experiment_name>' as "
                    "the prefix. If you genuinely need a different "
                    "location, add a `tests/` regression and document "
                    "the exception."
                ),
            )
        return HealthCheckResult(
            True,
            check_name,
            (f"training.output_dir='{path}' follows the experiments/results/<name> convention."),
            "info",
        )

    def check_identity_paths_agree(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Every identity path under ``experiments/results/`` names ONE experiment.

        :meth:`check_output_dir_convention` is a *prefix* test: it accepts
        ``experiments/results/<anything>`` and never asks whether ``<anything>``
        is this arm's own name. That blind spot is not hypothetical. It is
        recorded next door in :class:`spectramr.cli.profile_paths.ProfilePaths`,
        where a profiling run wrote checkpoints, metrics and provenance on top
        of the very arm it was measuring, and "the child's own config-health run
        reported the injected path as on-convention", leaving a profiling run
        and a real run indistinguishable after the fact.

        This is deliberately a SIBLING check rather than a fourth clause inside
        that one (non-negotiable 17): the prefix rule and the agreement rule
        fail for different reasons, quote different keys, and want different
        fix hints.

        **What it can and cannot see.** A single config is all this checker may
        read -- ``__init__`` takes no ``config_path`` by design -- so the
        question it can answer is *internal* consistency: an arm whose
        ``training.output_dir`` says ``foo`` while its checkpoints, logs and
        metrics say ``bar`` splits its own artifacts across two trees. It cannot
        see that two DIFFERENT arms name the same tree; that needs corpus
        knowledge and is owned by ``scripts/ci/check_identity_paths_unique.py``.

        **Scope.** Only values under ``experiments/results/`` are compared. A
        path left CWD-relative is a different defect with its own issue: over
        660 corpus arms, ``checkpoint.checkpoint_dir`` resolves on-convention
        for 384, inherits the ``./checkpoints`` default for 201 and is declared
        off-convention for 75 -- so 276 name no experiment at all there, and
        227 resolve to the exact literal ``./checkpoints``. Folding that in
        would fire on 42% of the corpus and get this check baselined and
        ignored.

        **What ``checkpoint.checkpoint_dir`` is, precisely.** It is not the
        happy-path checkpoint destination, and this check does not claim it is.
        ``CheckpointDirector`` -- the primary writer, at every save site in
        :mod:`spectramr.pipelines.training_loop` -- derives its directory from
        ``training.output_dir`` and never reads this key. The key is read by
        ``CheckpointService``, which the loop reaches only from the
        ``except Exception as director_err`` branch guarding the epoch save,
        and by ``CheckpointService.__init__``, which ``mkdir``s it on every run
        of every arm. So a disagreement here is a real split -- the directory
        is created, and it is where a checkpoint lands the moment the director
        raises -- but the bulk of a healthy run follows ``training.output_dir``.

        Severity is ``warning``: the arm trains, so this is not fatal, but
        ``audit`` is ``--strict`` and warnings exit 2 (non-negotiable 4).
        """
        check_name = "identity_paths_agree"
        by_segment: dict[str, list[str]] = {}
        for dotted in IDENTITY_PATHS:
            value = self._resolve_dotted(config, tuple(dotted.split(".")))
            if value is None:
                continue
            text = str(value).strip()
            if not text.startswith(RESULTS_PREFIX):
                continue
            rest = text[len(RESULTS_PREFIX) :].strip("/")
            if not rest:
                continue
            by_segment.setdefault(rest.split("/")[0], []).append(dotted)

        if len(by_segment) < 2:
            named = next(iter(by_segment), None)
            return HealthCheckResult(
                True,
                check_name,
                (
                    f"every identity path under {RESULTS_PREFIX} names '{named}'."
                    if named is not None
                    else f"no identity path points under {RESULTS_PREFIX}."
                ),
                "info",
            )

        rendered = "; ".join(
            f"'{segment}' <- {', '.join(sorted(keys))}"
            for segment, keys in sorted(by_segment.items())
        )
        return HealthCheckResult(
            False,
            check_name,
            (
                f"this arm's identity paths name {len(by_segment)} different "
                f"experiments, so its artifacts split across {len(by_segment)} "
                f"trees under {RESULTS_PREFIX}: {rendered}. Whichever segment is "
                f"wrong, part of this run lands in another arm's directory, "
                f"where nothing distinguishes it from that arm's own outputs "
                f"afterwards."
            ),
            "warning",
            category="identity_paths_disagree",
            yaml_keys=sorted(key for keys in by_segment.values() for key in keys),
            fix_hint=(
                f"Point every listed key at the same "
                f"{RESULTS_PREFIX}<experiment_name>. If the split is deliberate, "
                f"add a tests/ regression saying so -- no reader can tell it "
                f"from a copy-paste that kept the template's name."
            ),
        )

    def check_epochs_max_iterations_mutex(
        self,
        config: TrainingSettings,
    ) -> HealthCheckResult:
        """Info: both ``epochs`` and ``max_iterations`` are set.

        The training loop uses ``max_iterations`` as the canonical
        cap; ``epochs`` is informational/decorative. Setting both is
        the codebase convention, but this check exists to surface the
        ambiguity in JSON audit reports for tooling. Severity is
        ``info`` so it doesn't count as a warning.
        """
        check_name = "epochs_max_iterations_mutex"
        training = getattr(config, "training", None)
        if training is None:
            return HealthCheckResult(True, check_name, "no training block", "info")
        epochs = getattr(training, "epochs", None)
        max_it = getattr(training, "max_iterations", None)
        epochs_set = epochs is not None and int(epochs) > 1
        max_it_set = max_it is not None and int(max_it) > 0
        if epochs_set and max_it_set:
            return HealthCheckResult(
                True,  # informational only — codebase convention sets both
                check_name,
                (
                    f"training.epochs={epochs} and training.max_iterations="
                    f"{max_it} both set; max_iterations wins (loop cap)."
                ),
                "info",
            )
        return HealthCheckResult(True, check_name, "no mutex conflict", "info")

    # ============================================================
    # v6.1 — paradigm-expansion audit hooks
    # ============================================================

    def check_concomitant_requires_low_field(self, config: TrainingSettings) -> HealthCheckResult:
        """Warn if concomitant correction is on but B0 is not low-field."""
        check_name = "concomitant_requires_low_field"
        physics = getattr(config, "physics", None)
        concomitant = getattr(physics, "concomitant", None) if physics else None
        if concomitant is None or not getattr(concomitant, "enabled", False):
            return HealthCheckResult(True, check_name, "concomitant disabled.", "info")
        b0 = getattr(physics, "field_strength", None)
        if b0 is None:
            return HealthCheckResult(
                False,
                check_name,
                (
                    "physics.concomitant.enabled=True but physics.field_strength is "
                    "unset. The concomitant correction scales as G²/(2 B0); without "
                    "B0 the operator cannot run."
                ),
                "error",
                category="silent_fallback",
                yaml_keys=["physics.field_strength", "physics.concomitant.enabled"],
            )
        if b0 > 0.5:
            return HealthCheckResult(
                True,
                check_name,
                (
                    f"Concomitant correction enabled at B0={b0} T. The Maxwell term "
                    f"only matters at ULF (B0 < 0.2 T); on a clinical scanner the "
                    f"correction is a no-op but adds compute."
                ),
                # passed=True with severity "warning" was invisible to the single-arm
                # audit (report.warnings needs `not passed`) and an ERROR(strict) in
                # the bulk run, which counted severities: an advisory, reported as one.
                "info",
                category="potentially_redundant",
                always_report=True,
            )
        return HealthCheckResult(
            True, check_name, "B0 is low-field — correction is appropriate.", "info"
        )

    def check_field_strength_declared(self, config: TrainingSettings) -> HealthCheckResult:
        """ULF YAMLs must declare an explicit field_strength."""
        check_name = "field_strength_declared"
        physics = getattr(config, "physics", None)
        b0 = getattr(physics, "field_strength", None) if physics else None
        # ``metadata`` is ``dict[str, Any] | None`` — ``getattr(dict, "tags")``
        # always returns the default, so the ULF detection was dead. Read the
        # key for dicts and fall back to attribute access for objects.
        meta = getattr(config, "metadata", None)
        tags_val = meta.get("tags") if isinstance(meta, dict) else getattr(meta, "tags", None)
        is_ulf = tags_val is not None and "ulf" in str(tags_val).lower()
        if is_ulf and b0 is None:
            return HealthCheckResult(
                False,
                check_name,
                "ULF-tagged experiment must declare physics.field_strength explicitly.",
                "error",
                category="silent_fallback",
                yaml_keys=["physics.field_strength"],
            )
        return HealthCheckResult(True, check_name, "field_strength declared or non-ULF.", "info")

    # ----------------------------------------------------------------
    # PMPS Phase-0 (Bloch-grounded acquisition encoder) — Tier-1 checks.
    # ----------------------------------------------------------------
    def check_bloch_grounded_requires_reference_panel(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """PMPS Phase-0: bloch_grounded conditioning requires a reference panel."""
        check_name = "bloch_grounded_requires_reference_panel"
        data = getattr(config, "data", None)
        mc = getattr(data, "multi_contrast", None) if data else None
        if mc is None or not getattr(mc, "bloch_grounded", False):
            return HealthCheckResult(True, check_name, "bloch_grounded disabled.", "info")
        panel = getattr(mc, "reference_panel", None)
        if panel is None:
            return HealthCheckResult(
                False,
                check_name,
                (
                    "data.multi_contrast.bloch_grounded=True but no "
                    "data.multi_contrast.reference_panel is configured. The "
                    "BlochSignalEncoder needs a reference-tissue panel to "
                    "embed the acquisition vector."
                ),
                "error",
                category="silent_fallback",
                yaml_keys=[
                    "data.multi_contrast.bloch_grounded",
                    "data.multi_contrast.reference_panel.tissues",
                ],
                fix_hint=(
                    "Add `data.multi_contrast.reference_panel.tissues: [gm, wm, csf]` at minimum."
                ),
            )
        return HealthCheckResult(True, check_name, "reference_panel configured.", "info")

    def check_acquisition_params_required_for_bloch_grounded(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """PMPS Phase-0: bloch_grounded conditioning requires acquisition_params."""
        check_name = "acquisition_params_required_for_bloch_grounded"
        data = getattr(config, "data", None)
        mc = getattr(data, "multi_contrast", None) if data else None
        if mc is None or not getattr(mc, "bloch_grounded", False):
            return HealthCheckResult(True, check_name, "bloch_grounded disabled.", "info")
        params = getattr(mc, "acquisition_params", None)
        if not params:
            return HealthCheckResult(
                False,
                check_name,
                (
                    "data.multi_contrast.bloch_grounded=True but "
                    "data.multi_contrast.acquisition_params is empty. The "
                    "BlochSignalEncoder needs (TE, TR, TI, FA, B0) per "
                    "contrast to produce embeddings."
                ),
                "error",
                category="silent_fallback",
                yaml_keys=[
                    "data.multi_contrast.bloch_grounded",
                    "data.multi_contrast.acquisition_params",
                ],
                fix_hint=(
                    "Populate `data.multi_contrast.acquisition_params` with "
                    "one entry per contrast (TE, TR, FA, B0 required)."
                ),
            )
        return HealthCheckResult(True, check_name, "acquisition_params populated.", "info")

    def check_concomitant_required_at_ulf(self, config: TrainingSettings) -> HealthCheckResult:
        """PMPS Phase-0: every acquisition below 0.5 T must set include_concomitant.

        Schema-level auto-set covers most cases; this check catches YAMLs
        that *explicitly* pin ``include_concomitant: false`` at a low
        field strength — silently dropping the 1/B0-scaling Maxwell term
        and so producing physically wrong synthesised k-space.
        """
        check_name = "concomitant_required_at_ulf"
        data = getattr(config, "data", None)
        mc = getattr(data, "multi_contrast", None) if data else None
        if mc is None:
            return HealthCheckResult(True, check_name, "no multi_contrast block.", "info")
        params = getattr(mc, "acquisition_params", None) or []
        offenders: list[str] = []
        for p in params:
            b0 = getattr(p, "B0", 3.0)
            ic = getattr(p, "include_concomitant", None)
            if b0 < 0.5 and ic is False:
                offenders.append(f"{getattr(p, 'name', '<unnamed>')}(B0={b0}T)")
        if offenders:
            return HealthCheckResult(
                False,
                check_name,
                (
                    "Acquisition(s) below 0.5 T have "
                    "include_concomitant=False, which silently drops the "
                    "Maxwell concomitant phase (1/B0-scaling). At ULF this "
                    "produces physically wrong synthesised k-space. "
                    f"Offenders: {', '.join(offenders)}."
                ),
                "error",
                category="silent_fallback",
                yaml_keys=[
                    "data.multi_contrast.acquisition_params[*].include_concomitant",
                    "data.multi_contrast.acquisition_params[*].B0",
                ],
                fix_hint=(
                    "Remove the explicit `include_concomitant: false` so "
                    "the schema auto-rule (true when B0 < 0.5 T) applies, "
                    "or set it explicitly to true for low-field arms."
                ),
            )
        return HealthCheckResult(
            True, check_name, "no ULF acquisitions silently drop concomitant.", "info"
        )

    # ----------------------------------------------------------------
    # PMPS Phase-1/2/3/4 — paradigm-specific Tier-1 checks.
    # ----------------------------------------------------------------
    def _training_mode(self, config: TrainingSettings) -> str:
        return str(getattr(getattr(config, "training", None), "training_mode", "")).lower()

    def check_tissue_diffusion_pretrain_requires_field_strength_conditioning(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """PMPS Phase-1: heterogeneous-corpus training needs field_strength conditioning."""
        check_name = "tissue_diffusion_pretrain_requires_field_strength_conditioning"
        if self._training_mode(config) != "tissue_diffusion_pretrain":
            return HealthCheckResult(True, check_name, "not tissue_diffusion_pretrain.", "info")
        training = config.training
        fsc = getattr(training, "field_strength_conditioning", True)
        if not fsc:
            return HealthCheckResult(
                False,
                check_name,
                (
                    "training.field_strength_conditioning=False but PMPS "
                    "Phase-1 trains on a heterogeneous corpus that spans "
                    "multiple B0. Disable only when the corpus has a "
                    "single field strength."
                ),
                "error",
                category="silent_fallback",
                yaml_keys=["training.field_strength_conditioning"],
                fix_hint="Set training.field_strength_conditioning: true.",
            )
        return HealthCheckResult(True, check_name, "field_strength_conditioning enabled.", "info")

    def check_corruption_calibration_requires_tissue_prior(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """PMPS Phase-2: corruption calibration needs a tissue-prior checkpoint."""
        check_name = "corruption_calibration_requires_tissue_prior"
        if self._training_mode(config) != "corruption_calibration":
            return HealthCheckResult(True, check_name, "not corruption_calibration.", "info")
        ckpt = getattr(config.training, "tissue_prior_checkpoint", None)
        if ckpt is None:
            return HealthCheckResult(
                False,
                check_name,
                (
                    "training.tissue_prior_checkpoint is unset. Phase-2 "
                    "MMD calibration cannot run without the frozen prior "
                    "from Phase 1."
                ),
                "error",
                category="silent_fallback",
                yaml_keys=["training.tissue_prior_checkpoint"],
                fix_hint=(
                    "Point training.tissue_prior_checkpoint at the "
                    "checkpoint emitted by stage_a_tissue_prior_pretrain.yaml."
                ),
            )
        return HealthCheckResult(True, check_name, "tissue_prior_checkpoint configured.", "info")

    def check_paired_synthesis_requires_prior_and_calibration(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """PMPS Phase-3: synthesis needs BOTH the prior and the calibration checkpoint."""
        check_name = "paired_synthesis_requires_prior_and_calibration"
        if self._training_mode(config) != "paired_synthesis":
            return HealthCheckResult(True, check_name, "not paired_synthesis.", "info")
        training = config.training
        missing = []
        if getattr(training, "tissue_prior_checkpoint", None) is None:
            missing.append("training.tissue_prior_checkpoint")
        if getattr(training, "corruption_calibration_checkpoint", None) is None:
            missing.append("training.corruption_calibration_checkpoint")
        if missing:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"Phase-3 paired synthesis requires both prior and "
                    f"corruption-calibration checkpoints. Missing: "
                    f"{', '.join(missing)}."
                ),
                "error",
                category="silent_fallback",
                yaml_keys=missing,
            )
        return HealthCheckResult(True, check_name, "both checkpoints configured.", "info")

    def check_data_efficiency_harness_settings_well_formed(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """PMPS Phase-4: data-efficiency harness needs settings + manifests."""
        check_name = "data_efficiency_harness_settings_well_formed"
        if self._training_mode(config) != "data_efficiency_harness":
            return HealthCheckResult(True, check_name, "not data_efficiency_harness.", "info")
        training = config.training
        settings = list(getattr(training, "settings", []) or [])
        if not settings:
            return HealthCheckResult(
                False,
                check_name,
                (
                    "training.settings is empty. The harness needs at "
                    "least one cell (e.g. R10, R10_S100)."
                ),
                "error",
                category="silent_fallback",
                yaml_keys=["training.settings"],
            )
        for field in ("real_paired_manifest", "synthetic_paired_manifest"):
            if getattr(training, field, None) is None:
                return HealthCheckResult(
                    False,
                    check_name,
                    f"training.{field} is unset.",
                    "error",
                    category="silent_fallback",
                    yaml_keys=[f"training.{field}"],
                )
        return HealthCheckResult(
            True,
            check_name,
            f"{len(settings)} settings configured; manifests present.",
            "info",
        )

    # ----------------------------------------------------------------
    # Phase-1 (VF-Residual Conformal Calibration) — Tier-1 checks.
    # ----------------------------------------------------------------
    def check_vf_residual_marker_matrix_well_conditioned(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """vf_residual requires a well-conditioned marker basis on disk."""
        check_name = "vf_residual_marker_matrix_well_conditioned"
        cert = getattr(config, "certification", None)
        conformal = getattr(cert, "conformal", None) if cert else None
        if conformal is None or not getattr(conformal, "enabled", False):
            return HealthCheckResult(True, check_name, "conformal disabled.", "info")
        if getattr(conformal, "score_fn", "") != "vf_residual":
            return HealthCheckResult(True, check_name, "score_fn is not vf_residual.", "info")
        path = getattr(conformal, "marker_basis_path", None)
        if path is None:
            return HealthCheckResult(
                False,
                check_name,
                "score_fn='vf_residual' requires certification.conformal.marker_basis_path.",
                "error",
                category="silent_fallback",
                yaml_keys=["certification.conformal.marker_basis_path"],
            )
        try:
            from spectramr.infrastructure.calibration import marker_basis_condition_number

            kappa = marker_basis_condition_number(path)
        except FileNotFoundError:
            return HealthCheckResult(
                False,
                check_name,
                f"Marker basis not found at {path}.",
                "error",
                category="missing_artefact",
                yaml_keys=["certification.conformal.marker_basis_path"],
                fix_hint=(
                    "Generate the marker basis with the marker-design script "
                    "and store as a complex .pt tensor of shape (N, k)."
                ),
            )
        except Exception as exc:
            return HealthCheckResult(
                False,
                check_name,
                f"Loading marker basis at {path} raised {type(exc).__name__}: {exc}.",
                "error",
                category="malformed_artefact",
                yaml_keys=["certification.conformal.marker_basis_path"],
            )
        if kappa > 1e6:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"Marker basis at {path} is ill-conditioned: "
                    f"cond(M^H M) = {kappa:.3e} > 1e6. The projector "
                    f"P_M = M(M^H M)^{{-1}}M^H will be numerically unstable."
                ),
                "error",
                category="ill_conditioned_basis",
                yaml_keys=["certification.conformal.marker_basis_path"],
                fix_hint=(
                    "Regenerate the basis with a higher minimum singular-value "
                    "threshold or reduce the marker count."
                ),
            )
        return HealthCheckResult(
            True,
            check_name,
            f"Marker basis well-conditioned: cond(M^H M) = {kappa:.3e}.",
            "info",
        )

    def check_vf_residual_exchangeability_precondition(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """vf_residual requires IID marker injection across samples.

        The exchangeability hypothesis of the Vovk theorem demands that
        the non-conformity scores be exchangeable across calibration +
        test samples. When markers are injected on a per-sample basis
        through the digital twin, that's automatic; if a YAML pins the
        same marker template for every sample (e.g. via a fixed RNG
        seed across the loader), exchangeability is broken.
        """
        check_name = "vf_residual_exchangeability_precondition"
        cert = getattr(config, "certification", None)
        conformal = getattr(cert, "conformal", None) if cert else None
        if conformal is None or not getattr(conformal, "enabled", False):
            return HealthCheckResult(True, check_name, "conformal disabled.", "info")
        if getattr(conformal, "score_fn", "") != "vf_residual":
            return HealthCheckResult(True, check_name, "score_fn is not vf_residual.", "info")
        # Best-effort: check the digital-twin marker-injection config when present.
        data = getattr(config, "data", None)
        marker_inj = None
        if data is not None:
            marker_inj = getattr(data, "marker_injection", None) or getattr(
                getattr(data, "digital_twin", None) or object(),
                "marker_injection",
                None,
            )
        if marker_inj is not None and getattr(marker_inj, "iid_across_samples", True) is False:
            return HealthCheckResult(
                False,
                check_name,
                (
                    "data.marker_injection.iid_across_samples=False "
                    "violates the exchangeability hypothesis required by "
                    "split-conformal calibration."
                ),
                "error",
                category="silent_fallback",
                yaml_keys=["data.marker_injection.iid_across_samples"],
                fix_hint="Set data.marker_injection.iid_across_samples=True.",
            )
        return HealthCheckResult(
            True,
            check_name,
            "Exchangeability precondition holds (or is unset, defaulting to IID).",
            "info",
        )

    # ----------------------------------------------------------------
    # Phase-2 (IB-VF / InfoNCE) — Tier-1 checks.
    # ----------------------------------------------------------------
    @staticmethod
    def _typed_training_block(config: TrainingSettings, attr: str, model_cls: type):
        """Fetch ``training.<attr>``, coercing a raw dict to its typed schema.

        ``ib_vf`` / ``twin_dps`` are accepted as extra dicts by
        :class:`TrainingStrategyConfigSchema` (they are ``Any``-typed to
        avoid an import cycle, like ``tto``), so callers may receive a plain
        ``dict``. Coercing it here lets the downstream checks read real
        values and prevents ``'dict' object has no attribute ...`` crashes
        (audit-2026-05-21).
        """
        block = getattr(getattr(config, "training", None), attr, None)
        if isinstance(block, dict):
            try:
                block = model_cls(**block)
            except Exception:
                return block  # individual checks fall back to getattr defaults
        return block

    def check_infonce_effective_batch_size_sufficient(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """data.batch_size * grad_accum + memory_bank_size >= K (InfoNCE tightness)."""
        check_name = "infonce_effective_batch_size_sufficient"
        if self._training_mode(config) != "ib_vf":
            return HealthCheckResult(True, check_name, "not ib_vf.", "info")
        from spectramr.config.schemas.training.vf_advanced import IBVFConfig

        ib = self._typed_training_block(config, "ib_vf", IBVFConfig)
        if ib is None:
            return HealthCheckResult(True, check_name, "ib_vf block absent.", "info")
        batch_size = int(
            getattr(
                getattr(getattr(config, "data", object()), "loader", None),
                "batch_size",
                1,
            )
        )
        grad_accum = int(
            getattr(
                getattr(getattr(config, "optimization", object()), "gradient", None),
                "accumulation_steps",
                1,
            )
        )
        bank = int(getattr(ib, "memory_bank_size", 0))
        K = int(getattr(ib, "infonce_negatives_K", 32))
        effective = batch_size * grad_accum + bank
        if effective < K:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"Effective negatives = batch_size({batch_size}) * "
                    f"grad_accum({grad_accum}) + memory_bank_size({bank}) "
                    f"= {effective}, but ib_vf.infonce_negatives_K = {K}. "
                    f"InfoNCE-tightness requires effective >= K [CPC, Theorem 1]."
                ),
                "error",
                category="insufficient_negatives",
                yaml_keys=[
                    "data.batch_size",
                    "optimization.gradient.accumulation_steps",
                    "training.ib_vf.memory_bank_size",
                    "training.ib_vf.infonce_negatives_K",
                ],
                fix_hint=(
                    "Raise data.batch_size, increase gradient_accumulation_steps, "
                    "or enable training.ib_vf.memory_bank_size > 0."
                ),
            )
        return HealthCheckResult(
            True,
            check_name,
            f"Effective negatives {effective} >= K={K}.",
            "info",
        )

    def check_ib_beta_positive(self, config: TrainingSettings) -> HealthCheckResult:
        """Warn if both IB weights are zero (strategy degenerates to plain VF)."""
        check_name = "ib_beta_positive"
        if self._training_mode(config) != "ib_vf":
            return HealthCheckResult(True, check_name, "not ib_vf.", "info")
        from spectramr.config.schemas.training.vf_advanced import IBVFConfig

        ib = self._typed_training_block(config, "ib_vf", IBVFConfig)
        if ib is None:
            return HealthCheckResult(True, check_name, "ib_vf block absent.", "info")
        if float(ib.beta_plus) <= 0 and float(ib.beta_minus) <= 0:
            return HealthCheckResult(
                False,
                check_name,
                (
                    "Both ib_vf.beta_plus and ib_vf.beta_minus are 0 — the "
                    "strategy degenerates to plain virtual-fiducial. Switch "
                    "training_mode to 'virtual_fiducial' instead."
                ),
                "warning",
                category="paradigm_degeneracy",
                yaml_keys=["training.ib_vf.beta_plus", "training.ib_vf.beta_minus"],
            )
        return HealthCheckResult(True, check_name, "IB weights non-trivial.", "info")

    def check_nav_encoder_input_is_dc_kspace(self, config: TrainingSettings) -> HealthCheckResult:
        """The navigator must read the DC k-space line."""
        check_name = "nav_encoder_input_is_dc_kspace"
        if self._training_mode(config) != "ib_vf":
            return HealthCheckResult(True, check_name, "not ib_vf.", "info")
        from spectramr.config.schemas.training.vf_advanced import IBVFConfig

        ib = self._typed_training_block(config, "ib_vf", IBVFConfig)
        if ib is None:
            return HealthCheckResult(True, check_name, "ib_vf block absent.", "info")
        src = getattr(ib, "navigator_source", "dc_kspace")
        if src not in ("dc_kspace", "central_line"):
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"training.ib_vf.navigator_source = {src!r}; only "
                    "'dc_kspace' or 'central_line' (equivalent for Cartesian) "
                    "are valid."
                ),
                "error",
                category="silent_fallback",
                yaml_keys=["training.ib_vf.navigator_source"],
            )
        return HealthCheckResult(True, check_name, f"navigator_source={src}.", "info")

    # ----------------------------------------------------------------
    # Phase-3 (Twin-Likelihood DPS) — Tier-1 checks.
    # ----------------------------------------------------------------
    def check_twin_dps_guidance_scales_finite_and_positive(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """λ_y, λ_M must be in (0, 1e3]."""
        check_name = "twin_dps_guidance_scales_finite_and_positive"
        if self._training_mode(config) != "twin_dps":
            return HealthCheckResult(True, check_name, "not twin_dps.", "info")
        from spectramr.config.schemas.training.vf_advanced import TwinDPSConfig

        td = self._typed_training_block(config, "twin_dps", TwinDPSConfig)
        if td is None:
            return HealthCheckResult(True, check_name, "twin_dps block absent.", "info")
        for name in ("lambda_y", "lambda_M"):
            val = float(getattr(td, name, 0.0))
            if not (0.0 < val <= 1e3):
                sev = "error" if val <= 0 else "warning"
                return HealthCheckResult(
                    False,
                    check_name,
                    f"twin_dps.{name}={val} is outside (0, 1e3].",
                    sev,
                    category="out_of_range",
                    yaml_keys=[f"training.twin_dps.{name}"],
                )
        return HealthCheckResult(True, check_name, "guidance scales OK.", "info")

    def check_phase_stego_basis_compatible(self, config: TrainingSettings) -> HealthCheckResult:
        """marker_template_path must exist and have compatible shape."""
        check_name = "phase_stego_basis_compatible_with_image_shape"
        if self._training_mode(config) != "twin_dps":
            return HealthCheckResult(True, check_name, "not twin_dps.", "info")
        from spectramr.config.schemas.training.vf_advanced import TwinDPSConfig

        td = self._typed_training_block(config, "twin_dps", TwinDPSConfig)
        if td is None:
            return HealthCheckResult(True, check_name, "twin_dps block absent.", "info")
        from pathlib import Path

        tmpl = getattr(td, "marker_template_path", None)
        if tmpl is None or not Path(str(tmpl)).exists():
            return HealthCheckResult(
                False,
                check_name,
                f"marker_template_path missing: {tmpl}.",
                "error",
                category="missing_artefact",
                yaml_keys=["training.twin_dps.marker_template_path"],
            )
        return HealthCheckResult(True, check_name, "marker template present.", "info")

    def check_chd_calibration_corpus_size(self, config: TrainingSettings) -> HealthCheckResult:
        """CHD calibration must have ≥ 200 calibration images for a meaningful DKW slack."""
        check_name = "chd_calibration_corpus_size_sufficient"
        cert = getattr(config, "certification", None)
        chd = getattr(cert, "chd", None) if cert else None
        if chd is None or not getattr(chd, "enabled", False):
            return HealthCheckResult(True, check_name, "CHD disabled.", "info")
        n = getattr(chd, "n_calibration", 0)
        if n < 200:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"certification.chd.n_calibration={n} is below the recommended "
                    f"floor (200). The DKW slack at delta=0.001 will be too wide to "
                    f"certify a useful beta."
                ),
                "warning",
                category="weak_calibration",
                yaml_keys=["certification.chd.n_calibration"],
            )
        return HealthCheckResult(True, check_name, "calibration corpus is sufficient.", "info")

    def check_pathology_test_set_size(self, config: TrainingSettings) -> HealthCheckResult:
        """PRC needs ≥ 100 pathology examples per class for a tight Hoeffding bound."""
        check_name = "pathology_test_set_size_sufficient"
        cert = getattr(config, "certification", None)
        prc = getattr(cert, "pathology_recall", None) if cert else None
        if prc is None or not getattr(prc, "enabled", False):
            # PRC default is enabled=False; only fire if the user opted in.
            if prc is None or not getattr(prc, "classes", None):
                return HealthCheckResult(True, check_name, "PRC not configured.", "info")
        # We can't read the manifest here cheaply; emit informational reminder.
        return HealthCheckResult(
            True,
            check_name,
            (
                f"PRC active for classes={getattr(prc, 'classes', [])}. The Hoeffding "
                f"bound at delta={getattr(prc, 'delta', 0.05)} requires ≥ 100 examples "
                f"per class — verify the manifest size before submitting."
            ),
            "info",
        )

    def check_dp_compatible_optimizer(self, config: TrainingSettings) -> HealthCheckResult:
        """DP-SGD requires a per-sample-clip-friendly optimizer (no momentum bias-state hacks)."""
        check_name = "dp_compatible_optimizer"
        # Heuristic: if training.federated.differential_privacy is on, recommend
        # a non-momentum optimizer or AdamW.
        training = getattr(config, "training", None)
        dp_on = False
        if training is not None:
            for k in ("federated", "dp", "differential_privacy"):
                sub = getattr(training, k, None)
                if sub and getattr(sub, "enabled", False):
                    dp_on = True
                    break
        if not dp_on:
            return HealthCheckResult(True, check_name, "DP-SGD disabled.", "info")
        opt = getattr(config, "optimization", None)
        opt_type = getattr(opt.optimizer, "type", "") if opt else ""
        if opt_type.lower() not in {"sgd", "adamw"}:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"DP-SGD active but optimizer_type={opt_type!r}. Use sgd or adamw "
                    f"for compatible per-sample clipping."
                ),
                "warning",
                category="dp_incompatibility",
                yaml_keys=["optimization.optimizer.type"],
            )
        return HealthCheckResult(True, check_name, "Optimizer is DP-compatible.", "info")

    def check_validation_badge_gates_complete(self, config: TrainingSettings) -> HealthCheckResult:
        """Validation badge must declare every required gate or it can never pass."""
        check_name = "validation_badge_gates_complete"
        cert = getattr(config, "certification", None)
        badge = getattr(cert, "badge", None) if cert else None
        if badge is None or not getattr(badge, "enabled", False):
            return HealthCheckResult(True, check_name, "validation badge disabled.", "info")
        required = {
            "iqm",
            "downstream",
            "hallucination",
            "conformal_coverage",
            "pathology_recall",
            "pac_bayes",
        }
        declared = set(getattr(badge, "gates", []) or [])
        missing = required - declared
        if missing:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"validation.badge.gates is missing required gates: "
                    f"{sorted(missing)}. The regulatory-package CLI will reject "
                    f"the badge as incomplete."
                ),
                "error",
                category="incomplete_badge",
                yaml_keys=["certification.badge.gates"],
            )
        return HealthCheckResult(True, check_name, "all required gates declared.", "info")

    def check_acceleration_adaptive_target_rate_consistent(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """Adaptive-mask target_rate should match base_acceleration."""
        check_name = "adaptive_target_rate_consistent"
        accel = getattr(config, "undersampling", None)
        adaptive = getattr(accel, "adaptive", None) if accel else None
        if adaptive is None or not getattr(adaptive, "enabled", False):
            return HealthCheckResult(True, check_name, "adaptive mask disabled.", "info")
        target = getattr(adaptive, "target_rate", 0.125)
        base = getattr(accel, "base_acceleration", 8.0)
        expected = 1.0 / max(base, 1e-9)
        if abs(target - expected) > 0.05:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"acceleration.adaptive.target_rate={target} disagrees with "
                    f"base_acceleration={base} (expected target_rate≈{expected:.3f}). "
                    f"The density penalty will pull toward an inconsistent budget."
                ),
                "warning",
                category="mask_budget_mismatch",
                yaml_keys=[
                    "acceleration.adaptive.target_rate",
                    "acceleration.base_acceleration",
                ],
            )
        return HealthCheckResult(True, check_name, "target_rate is consistent.", "info")

    def check_pilot_hardware_constraints_sane(self, config: TrainingSettings) -> HealthCheckResult:
        """Slew rate / max gradient must be physically feasible."""
        check_name = "pilot_hardware_constraints_sane"
        acq = getattr(config, "acquisition", None)
        codesign = getattr(acq, "codesign", None) if acq else None
        if codesign is None or not getattr(codesign, "enabled", False):
            return HealthCheckResult(True, check_name, "codesign disabled.", "info")
        c = codesign.constraints
        if c.slew_rate_max_T_per_m_per_s > 1000.0:
            return HealthCheckResult(
                False,
                check_name,
                (
                    f"Slew rate {c.slew_rate_max_T_per_m_per_s} T/m/s exceeds typical "
                    f"clinical hardware (200 T/m/s). PILOT solver will diverge."
                ),
                "error",
                category="hardware_infeasible",
            )
        return HealthCheckResult(True, check_name, "constraints are physically sensible.", "info")

    @staticmethod
    def _resolve_marker_basis_path(config: TrainingSettings) -> str | None:
        """Resolve a declared marker-basis tensor path across known schema homes."""
        for chain in (
            ("physics", "virtual_fiducial", "marker_path"),
            ("certification", "conformal", "marker_basis_path"),
        ):
            obj: object | None = config
            for attr in chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, str) and obj:
                return obj
        return None

    @staticmethod
    def _resolve_checkpoint_paths(config: TrainingSettings) -> list[tuple[str, str]]:
        """Collect (yaml_key, path) for every declared checkpoint / basis / template."""
        out: list[tuple[str, str]] = []
        candidates = (
            ("checkpoint.resume_from", ("checkpoint", "resume_from")),
            ("model.checkpoint_path", ("model", "checkpoint_path")),
            (
                "certification.conformal.marker_basis_path",
                ("certification", "conformal", "marker_basis_path"),
            ),
            (
                "certification.conformal.pretrained_reconstructor_checkpoint",
                ("certification", "conformal", "pretrained_reconstructor_checkpoint"),
            ),
            (
                "training.twin_dps.marker_template_path",
                ("training", "twin_dps", "marker_template_path"),
            ),
            (
                "training.twin_dps.pretrained_score_checkpoint",
                ("training", "twin_dps", "pretrained_score_checkpoint"),
            ),
        )
        for key, chain in candidates:
            obj: object | None = config
            for attr in chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, str) and obj:
                out.append((key, obj))
            elif isinstance(obj, (list, tuple)):
                out.extend((key, p) for p in obj if isinstance(p, str) and p)
        return out

    def check_marker_subspace_conditioning(
        self, config: TrainingSettings, kappa_max: float = 1e4
    ) -> HealthCheckResult:
        """Tier-1 (VF plan I-5): marker subspace conditioning κ(M) ≤ kappa_max.

        An ill-conditioned marker basis (κ ≫ 1) makes the VF-residual projection
        P_M numerically unstable and inflates every downstream conformal interval.
        Fires only when a marker-basis tensor is declared *and* present on disk;
        synthetic-injection arms declare no basis file and info-skip.
        """
        name = "marker_subspace_conditioning"
        path = self._resolve_marker_basis_path(config)
        if path is None:
            return HealthCheckResult(
                passed=True,
                check_name=name,
                message="No marker-basis tensor declared; check skipped.",
                severity="info",
            )
        from pathlib import Path

        if not Path(path).exists():
            return HealthCheckResult(
                passed=True,
                check_name=name,
                message=f"Marker basis '{path}' absent locally; conditioning not "
                "evaluated (presence is gated by checkpoint_existence).",
                severity="info",
                yaml_keys=["certification.conformal.marker_basis_path"],
            )
        try:
            import torch

            loaded = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(loaded, dict):
                loaded = loaded.get("basis", loaded.get("marker"))
            if loaded is None or not torch.is_tensor(loaded) or loaded.numel() == 0:
                raise ValueError("loaded object is not a non-empty tensor")
            mat = loaded.reshape(-1, loaded.shape[-1]).to(torch.float32)
            sigma = torch.linalg.svdvals(mat)
            kappa = float((sigma[0] / sigma[-1].clamp_min(1e-12)).item())
        except (OSError, RuntimeError, ValueError, EOFError, KeyError) as exc:
            return HealthCheckResult(
                passed=False,
                check_name=name,
                message=f"Could not evaluate marker conditioning for '{path}': {exc}",
                severity="warning",
                yaml_keys=["certification.conformal.marker_basis_path"],
                fix_hint="Save the marker basis as a 2-D tensor [N, N_M] via torch.save.",
            )
        passed = kappa <= kappa_max
        return HealthCheckResult(
            passed=passed,
            check_name=name,
            message=f"κ(M) = {kappa:.2e} (threshold {kappa_max:.2e})",
            severity="info" if passed else "error",
            yaml_keys=["certification.conformal.marker_basis_path"],
            fix_hint=(
                None
                if passed
                else (
                    "Marker subspace ill-conditioned: use an orthonormal basis "
                    "(random unitary), widen marker support, or reduce N_M."
                )
            ),
        )

    def check_checkpoint_existence(self, config: TrainingSettings) -> HealthCheckResult:
        """Tier-1 (VF plan CW-2): declared checkpoint / basis / template paths exist.

        Warning severity so a plain ``spectramr audit`` does not hard-fail on a
        cluster path that is legitimately absent locally; ``--strict`` (the
        smoke / dispatch wrapper) escalates it to a hard gate — the intended
        pre-flight that stops an arm whose upstream artefact was never built.
        """
        name = "checkpoint_existence"
        from pathlib import Path

        declared = self._resolve_checkpoint_paths(config)
        if not declared:
            return HealthCheckResult(
                passed=True,
                check_name=name,
                message="No checkpoint / basis dependency declared.",
                severity="info",
            )
        missing = [(k, p) for k, p in declared if not Path(p).exists()]
        if not missing:
            return HealthCheckResult(
                passed=True,
                check_name=name,
                message=f"All {len(declared)} declared checkpoint/basis path(s) present.",
                severity="info",
            )

        # Campaign-dependency deferral (VF Option A, 2026-06-16). A downstream
        # eval/calibration/DPS arm may declare ``checkpoint.produced_by_arm`` —
        # the upstream campaign arm that builds its checkpoint(s). When set, a
        # still-absent *campaign-artefact-rooted* checkpoint is a deferred
        # dependency (the campaign runner builds the producer first), so the
        # standalone --strict pre-flight info-passes instead of hard-failing.
        # The deferral is deliberately narrow (CLAUDE.md #15 — validate, don't
        # rubber-stamp): it does NOT cover non-artefact paths (a typo'd local
        # path) or precomputed data artefacts (marker basis/template tensors,
        # which no training arm produces). The real existence gate still fires
        # at actual checkpoint-load time.
        produced_by = getattr(getattr(config, "checkpoint", None), "produced_by_arm", None)
        produced_by = produced_by.strip() if isinstance(produced_by, str) else None
        if produced_by:
            artefact_roots = ("experiments/active/", "experiments/results/")

            def _is_checkpoint_key(key: str) -> bool:
                return "checkpoint" in key or "resume_from" in key

            def _is_artefact_rooted(path: str) -> bool:
                norm = path.replace("\\", "/").lstrip("./")
                return any(r in norm for r in artefact_roots)

            deferred = [
                (k, p) for k, p in missing if _is_checkpoint_key(k) and _is_artefact_rooted(p)
            ]
            remaining = [m for m in missing if m not in deferred]
            invalid_defer = [(k, p) for k, p in remaining if _is_checkpoint_key(k)]
            data_missing = [(k, p) for k, p in remaining if not _is_checkpoint_key(k)]

            if invalid_defer:
                return HealthCheckResult(
                    passed=False,
                    check_name=name,
                    message=(
                        f"checkpoint.produced_by_arm='{produced_by}' only defers "
                        "campaign-artefact checkpoints rooted under experiments/active/ "
                        "or experiments/results/; these declared path(s) are not: "
                        + ", ".join(f"{k}={p}" for k, p in invalid_defer)
                    ),
                    severity="error",
                    yaml_keys=[k for k, _ in invalid_defer],
                    fix_hint="Point the checkpoint at the producer arm's campaign-artefact "
                    "output path, or drop produced_by_arm and provide the file.",
                )
            if data_missing:
                return HealthCheckResult(
                    passed=False,
                    check_name=name,
                    message=(
                        "Missing precomputed data artefact(s) (NOT produced by a training "
                        f"arm, so not covered by produced_by_arm='{produced_by}'): "
                        + ", ".join(f"{k}={p}" for k, p in data_missing)
                    ),
                    severity="warning",
                    yaml_keys=[k for k, _ in data_missing],
                    fix_hint="Generate the precomputed artefact (e.g. marker basis/template "
                    "tensor) and commit it; produced_by_arm does not build data artefacts.",
                )
            return HealthCheckResult(
                passed=True,
                check_name=name,
                message=(
                    f"{len(deferred)} checkpoint(s) deferred to upstream campaign "
                    f"producer '{produced_by}' (built at campaign dispatch, gated at "
                    "checkpoint-load): " + ", ".join(f"{k}={p}" for k, p in deferred)
                ),
                severity="info",
                yaml_keys=[k for k, _ in deferred],
            )

        return HealthCheckResult(
            passed=False,
            check_name=name,
            message="Missing declared artefact(s): " + ", ".join(f"{k}={p}" for k, p in missing),
            severity="warning",
            yaml_keys=[k for k, _ in missing],
            fix_hint="Produce the upstream artefact first (campaign dependency "
            "graph), declare checkpoint.produced_by_arm if an upstream campaign arm "
            "builds it, or remove the declaration; --strict gates dispatch on this.",
        )

    @staticmethod
    def _conditioning_source_error(
        sources: list[str],
        registered: set[str],
        supported: set[str] | None,
    ) -> str | None:
        """Pure decision logic for the conditioning guard (testable alone).

        ``supported is None`` ⇒ the strategy class could not be resolved, so
        only the registered-source check runs. An empty ``supported`` set ⇒ the
        resolved strategy consumes no conditioning (silent no-op).
        """
        unknown = [s for s in sources if s not in registered]
        if unknown:
            return (
                f"unknown conditioning source(s) {unknown}; registered encoders: "
                f"{sorted(registered)}. Register one via @register_conditioner or "
                "remove the source."
            )
        if supported is not None:
            if not supported:
                return (
                    "model.conditioning.enabled but the training strategy consumes "
                    "no conditioning (_SUPPORTED_CONDITION_SOURCES is empty) — "
                    "enabling it here is a silent no-op (CLAUDE.md #9). Wire the "
                    "strategy to consume the context, or disable conditioning."
                )
            missing = [s for s in sources if s not in supported]
            if missing:
                return (
                    f"the training strategy does not consume conditioning source(s) "
                    f"{missing}; it supports {sorted(supported)}. Those sources would "
                    "be silently ignored at runtime."
                )
        return None

    def check_conditioning_sources_supported(self, config: TrainingSettings) -> HealthCheckResult:
        """Tier-1: ``model.conditioning`` sources must be registered AND consumed.

        Guards the silent-no-op surface: a YAML that enables conditioning on a
        strategy which never populates the context, or names an unregistered
        source, fails at audit time instead of being ignored at runtime.
        """
        cond = getattr(getattr(config, "model", None), "conditioning", None)
        if cond is None or not getattr(cond, "enabled", False):
            return HealthCheckResult(
                passed=True,
                check_name="conditioning_sources_supported",
                message="model.conditioning disabled — skipped.",
                severity="info",
            )
        sources = list(getattr(cond, "sources", []) or [])
        from spectramr.models.conditioning.encoders import _CONDITIONER_REGISTRY

        registered = set(_CONDITIONER_REGISTRY)
        supported: set[str] | None
        try:
            from spectramr.infrastructure.training.strategy_factory import (
                TrainingStrategyFactory,
            )

            strat_cls = TrainingStrategyFactory.get_strategy_class(config)
            supported = set(getattr(strat_cls, "_SUPPORTED_CONDITION_SOURCES", ()) or ())
        except Exception:
            supported = None  # unresolved — only source-validity is enforced

        err = self._conditioning_source_error(sources, registered, supported)
        if err:
            return HealthCheckResult(
                passed=False,
                check_name="conditioning_sources_supported",
                message=err,
                severity="error",
                category="conditioning",
                yaml_keys=["model.conditioning.sources"],
            )
        return HealthCheckResult(
            passed=True,
            check_name="conditioning_sources_supported",
            message=(
                f"conditioning sources {sources} are registered and supported by "
                "the strategy (runtime propagation enforced by the mechanism-fires "
                "guard in ConditioningMixin._apply_input_conditioning)."
            ),
            severity="info",
        )

    # ------------------------------------------------------------------
    # SPECTRA hardware backend checks (implementation plan Workstream C, C4)
    #
    # All three no-op (passed=True, severity="info") when no SPECTRA backend
    # block is configured, so non-SPECTRA configs are entirely unaffected.
    # ------------------------------------------------------------------

    def check_spectra_backend_schema(self, config: Any) -> HealthCheckResult:
        """Guard: a SPECTRA backend must be declared via the typed schema block.

        Fires (error) when something selects the spectra backend but the
        ``backend_acceleration`` block is absent, or when the block is present
        but its ``backend`` is not ``"spectra"`` — the "selected without schema"
        invariant. This is the config-time equivalent of the Tier-0
        ``extra="forbid"`` guard for the *value* (Tier-0 already rejects unknown
        keys; this rejects a mis-selected/missing backend).
        """
        block = getattr(config, "backend_acceleration", None)

        # Detect a stray backend selection elsewhere (e.g. acceleration.backend
        # == "spectra" pointed at the wrong block) without the typed schema.
        accel = getattr(config, "undersampling", None)
        stray = getattr(accel, "backend", None) if accel is not None else None
        if block is None:
            if stray == "spectra":
                return HealthCheckResult(
                    passed=False,
                    check_name="spectra_backend_schema",
                    message=(
                        "The SPECTRA backend is selected (acceleration.backend="
                        "'spectra') but the typed 'backend_acceleration' block is "
                        "missing."
                    ),
                    severity="error",
                    category="backend_acceleration",
                    yaml_keys=["acceleration.backend", "backend_acceleration"],
                    fix_hint=(
                        "Add a 'backend_acceleration:' block with backend: spectra "
                        "(see spectramr.config.schemas.spectra)."
                    ),
                )
            return HealthCheckResult(
                passed=True,
                check_name="spectra_backend_schema",
                message="No SPECTRA backend configured (not applicable).",
                severity="info",
                category="backend_acceleration",
            )

        backend = getattr(block, "backend", None)
        if backend != "spectra":
            return HealthCheckResult(
                passed=False,
                check_name="spectra_backend_schema",
                message=(
                    f"backend_acceleration.backend={backend!r} is not 'spectra'; "
                    "this block only configures the SPECTRA backend."
                ),
                severity="error",
                category="backend_acceleration",
                yaml_keys=["backend_acceleration.backend"],
                fix_hint="Set backend_acceleration.backend: spectra.",
            )

        return HealthCheckResult(
            passed=True,
            check_name="spectra_backend_schema",
            message="SPECTRA backend schema present and well-formed.",
            severity="info",
            category="backend_acceleration",
        )

    def check_spectra_precision_envelope(self, config: Any) -> HealthCheckResult:
        """Guard: reject FP8 *output* precision for a diagnostic recon task.

        Encodes the A7 numerics discipline (implementation plan Workstream C):
        FP8-E4M3 alone falls below the diagnostic PSNR/SSIM envelope, so it is an
        inner-loop precision, never an output precision. When the SPECTRA backend
        is configured with ``pe_mode_policy == fp8`` for a reconstruction-class
        task, this is an error.
        """
        block = getattr(config, "backend_acceleration", None)
        if block is None:
            return HealthCheckResult(
                passed=True,
                check_name="spectra_precision_envelope",
                message="No SPECTRA backend configured (not applicable).",
                severity="info",
                category="backend_acceleration",
            )

        policy = getattr(block, "pe_mode_policy", None)
        policy_val = getattr(policy, "value", policy)  # Enum or raw str

        # Diagnostic reconstruction-class tasks: reading the training mode and
        # the model type (either signal is sufficient).
        training = getattr(config, "training", None)
        mode = getattr(training, "training_mode", None) if training is not None else None
        mode_val = getattr(mode, "value", mode)
        model = getattr(config, "model", None)
        model_type = getattr(model, "model_type", None) if model is not None else None
        is_diagnostic_recon = str(mode_val).lower() == "reconstruction" or str(
            model_type
        ).lower() in {
            "modl",
            "deep_cascade",
            "reconstruction",
            "varnet",
            "unrolled_reconstruction",
        }

        if policy_val == "fp8" and is_diagnostic_recon:
            return HealthCheckResult(
                passed=False,
                check_name="spectra_precision_envelope",
                message=(
                    "pe_mode_policy=fp8 is below the diagnostic-fidelity envelope "
                    f"for a reconstruction task (model={model_type!r}). FP8 is an "
                    "inner-loop precision, not an output precision."
                ),
                severity="error",
                category="backend_acceleration",
                yaml_keys=["backend_acceleration.pe_mode_policy", "model.model_type"],
                fix_hint=(
                    "Raise the output-precision floor to BF16 or FP32 "
                    "(pe_mode_policy: bf16 | fp32 | mixed)."
                ),
            )

        return HealthCheckResult(
            passed=True,
            check_name="spectra_precision_envelope",
            message=f"SPECTRA precision policy '{policy_val}' meets the fidelity envelope.",
            severity="info",
            category="backend_acceleration",
        )

    def check_spectra_gridding_sized(self, config: Any) -> HealthCheckResult:
        """Guard: the NUFFT gridding kernel must fit the accumulator depth.

        From the A4 sizing model: each non-uniform sample scatters into a
        ``J**spatial_dims`` neighbourhood, so the engine must coalesce at least
        that many in-flight grid addresses. ``J**D`` must not exceed the
        synthesized accumulator depth bound. Spatial dimensionality is taken
        from ``data.sampling.patch_size`` length (default 2-D).
        """
        block = getattr(config, "backend_acceleration", None)
        if block is None:
            return HealthCheckResult(
                passed=True,
                check_name="spectra_gridding_sized",
                message="No SPECTRA backend configured (not applicable).",
                severity="info",
                category="backend_acceleration",
            )

        kernel_width = getattr(block, "gridding_kernel_width", None)
        if kernel_width is None:
            kernel_width = 4  # schema default

        data = getattr(config, "data", None)
        patch_size = data.sampling.patch_size if data is not None else None
        spatial_dims = len(patch_size) if isinstance(patch_size, (list, tuple)) else 2
        spatial_dims = max(2, spatial_dims)

        # Synthesized accumulator-depth bound (A4): the gridding engine provides
        # banked accumulators sized to a fixed J^2_max footprint. The published
        # first-cut engine targets J up to 6 in 2-D (J^2 = 36), so the bound is
        # 6**spatial_dims — generous in 2-D, tight in 3-D.
        accumulator_depth_bound = 6**spatial_dims
        required_depth = kernel_width**spatial_dims

        if required_depth > accumulator_depth_bound:
            return HealthCheckResult(
                passed=False,
                check_name="spectra_gridding_sized",
                message=(
                    f"gridding_kernel_width={kernel_width} over {spatial_dims}-D "
                    f"requires {required_depth} banked accumulators, exceeding the "
                    f"synthesized bound of {accumulator_depth_bound}."
                ),
                severity="error",
                category="backend_acceleration",
                yaml_keys=[
                    "backend_acceleration.gridding_kernel_width",
                    "data.sampling.patch_size",
                ],
                fix_hint=(
                    "Reduce gridding_kernel_width (J) so J**spatial_dims <= "
                    f"{accumulator_depth_bound}, or lower the spatial dimensionality."
                ),
            )

        return HealthCheckResult(
            passed=True,
            check_name="spectra_gridding_sized",
            message=(
                f"SPECTRA gridding sized OK: J**{spatial_dims}={required_depth} "
                f"<= {accumulator_depth_bound}."
            ),
            severity="info",
            category="backend_acceleration",
        )

    def check_tta_adapter_only(self, config: Any) -> HealthCheckResult:
        """Guard: a SPECTRA-TTA config must run the spectra_tta strategy with a
        frozen backbone.

        TTA is the only training-flavoured SPECTRA paradigm and its defensibility
        rests on adapting *only* norm/adapter parameters: freezing the backbone is
        what collapses the train/infer memory multiplier from ``|W|`` to
        ``|W_adapter|`` so adaptation fits the inference datapath (plan §D). This
        check fires when a ``training.spectra_tta`` block is present and asserts
        (a) the strategy actually resolves to spectra_tta, and (b) freeze_backbone
        is not disabled.
        """
        training_cfg = getattr(config, "training", None)
        if training_cfg is None:
            return HealthCheckResult(
                passed=True,
                check_name="tta_adapter_only",
                message="No training config (not applicable).",
                severity="info",
                category="spectra_tta",
            )

        tta_block = getattr(training_cfg, "spectra_tta", None)
        mode = getattr(training_cfg, "training_mode", None)
        strategy_class = getattr(training_cfg, "strategy_class", None) or ""
        selects_tta = (
            tta_block is not None
            or mode == "spectra_tta"
            or "spectra_tta" in strategy_class.lower()
            or "spectratesttimeadaptation" in strategy_class.lower()
        )
        if not selects_tta:
            return HealthCheckResult(
                passed=True,
                check_name="tta_adapter_only",
                message="SPECTRA TTA not configured (not applicable).",
                severity="info",
                category="spectra_tta",
            )

        # (a) the strategy must resolve to spectra_tta
        if not (
            mode == "spectra_tta"
            or "spectra_tta" in strategy_class.lower()
            or "spectratesttimeadaptation" in strategy_class.lower()
        ):
            return HealthCheckResult(
                passed=False,
                check_name="tta_adapter_only",
                message=(
                    "A training.spectra_tta block is present but the strategy does "
                    f"not select the SPECTRA TTA paradigm (training_mode={mode!r}, "
                    f"strategy_class={strategy_class!r})."
                ),
                severity="error",
                category="spectra_tta",
                yaml_keys=[
                    "training.training_mode",
                    "training.strategy_class",
                    "training.spectra_tta",
                ],
                fix_hint=(
                    "Set training_mode: spectra_tta (or strategy_class to the "
                    "SpectraTestTimeAdaptationStrategy path)."
                ),
            )

        # (b) the backbone must stay frozen — that is the bounded-memory guarantee.
        # The block arrives as a plain dict (TrainingStrategyConfigSchema.spectra_tta
        # is an Any passthrough, like tto) OR as a coerced SpectraTTAConfig object,
        # so read both shapes.
        if isinstance(tta_block, dict):
            freeze = tta_block.get("freeze_backbone", True)
        else:
            freeze = getattr(tta_block, "freeze_backbone", True)
        if freeze is False:
            return HealthCheckResult(
                passed=False,
                check_name="tta_adapter_only",
                message=(
                    "training.spectra_tta.freeze_backbone is False: adapting the full "
                    "backbone blows up the train/infer memory multiplier (gradients + "
                    "Adam moments scale with |W|, not |W_adapter|) and will not fit the "
                    "inference datapath."
                ),
                severity="error",
                category="spectra_tta",
                yaml_keys=["training.spectra_tta.freeze_backbone"],
                fix_hint="Set training.spectra_tta.freeze_backbone: true (debug-only when false).",
            )

        return HealthCheckResult(
            passed=True,
            check_name="tta_adapter_only",
            message="SPECTRA TTA configured with a frozen backbone (adapter-only).",
            severity="info",
            category="spectra_tta",
        )

    def check_b0_field_metric_requires_hz_model(
        self, config: "TrainingSettings"
    ) -> HealthCheckResult:
        """The static half of the b0_field_rmse units lock (pitfall #16).

        ``b0_field_rmse`` is an *absolute Hz* metric: grading anything that is
        not an Hz field against a real B0 is the bssfp_peak_finder facade (it
        returns a corrected image, not a field). So if the config requests
        ``b0_field_rmse`` (or its ``b0_rmse_hz`` alias), the chosen model MUST
        declare ``output_field_units == "Hz"``. The runtime guard
        (``field_comparability``) is the second lock; together they make the
        Hz-vs-Hz invariant un-bypassable by a later config edit.
        """
        check_name = "b0_field_metric_requires_hz_model"
        validation = getattr(config, "validation", None)
        metrics = (validation.scoring.compute if validation else None) or []
        requested = {m.lower() for m in metrics if isinstance(m, str)}
        if not (requested & {"b0_field_rmse", "b0_rmse_hz"}):
            return HealthCheckResult(
                True,
                check_name,
                "b0_field_rmse not requested — units lock not applicable.",
                "info",
            )
        model = getattr(config, "model", None)
        model_type = getattr(model, "model_type", None)
        units = None
        try:
            from spectramr.models.registry import get_model_capabilities

            caps = get_model_capabilities(model_type) if model_type else None
            units = getattr(caps, "output_field_units", None) if caps else None
        except Exception:  # registry miss must not crash the audit
            units = None
        if units == "Hz":
            return HealthCheckResult(
                True,
                check_name,
                f"model {model_type!r} declares output_field_units='Hz' for b0_field_rmse.",
                "info",
                category="b0_field_metric_units",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                f"validation.metrics requests 'b0_field_rmse' (an absolute Hz "
                f"metric) but model {model_type!r} does not declare "
                f"output_field_units='Hz' (got {units!r}). Grading a non-Hz / "
                f"image output as a B0 field is the bssfp_peak_finder facade "
                f"(pitfall #16)."
            ),
            "error",
            category="b0_field_metric_units",
            yaml_keys=["validation.metrics", "model.model_type"],
            fix_hint=(
                "Use a model that declares output_field_units='Hz' (e.g. "
                "bssfp_b0_regressor), or drop b0_field_rmse from validation.scoring."
            ),
        )

    def check_dds_requires_score_model(self, config: "TrainingSettings") -> HealthCheckResult:
        """DDS is a score-model posterior sampler — guard against misuse (step 2).

        ``inference_sampler: dds`` resolves to ``DDSReconSampler``, which needs a
        score / eps-prediction diffusion model exposing ``_model_forward`` and a
        variance schedule (``alphas_cumprod`` / ``sqrt_alphas_cumprod``). Cold
        diffusion (``kspace_cold_diffusion``) has neither, and non-diffusion
        paradigms are not posterior-sampleable at all — selecting ``dds`` there
        fails loud at runtime, so reject it at config time (pitfalls #15/#16).
        """
        check_name = "dds_requires_score_model"
        training = getattr(config, "training", None)
        diffusion = getattr(training, "diffusion", None)
        sampler = getattr(diffusion, "inference_sampler", None) or getattr(
            diffusion, "sampler", None
        )
        posterior_samplers = {"dds", "dynamic_dps"}
        if not (isinstance(sampler, str) and sampler.lower() in posterior_samplers):
            return HealthCheckResult(
                True,
                check_name,
                "inference_sampler is not a score-model posterior sampler "
                "(dds / dynamic_dps) — guard not applicable.",
                "info",
            )
        training_mode = str(getattr(training, "training_mode", "") or "").lower()
        model_type = str(getattr(getattr(config, "model", None), "model_type", "") or "").lower()
        # Score/eps-prediction paradigm: the DDPM-style 'diffusion' mode, or any
        # model that names itself a score model. Cold diffusion is explicitly NOT
        # score-based (deterministic degradation, no eps + alphas_cumprod).
        is_score = (
            (
                training_mode in {"diffusion", "edm", "score_field_tomography"}
                or "score" in model_type
            )
            and "cold" not in training_mode
            and "cold" not in model_type
        )
        if is_score:
            return HealthCheckResult(
                True,
                check_name,
                f"inference_sampler={sampler.lower()!r} on a score paradigm "
                f"(training_mode={training_mode!r}, model={model_type!r}).",
                "info",
                category="sampler_model_compat",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                f"inference_sampler={sampler.lower()!r} requires a score/eps-prediction "
                f"diffusion model (exposing _model_forward + alphas_cumprod), but "
                f"training_mode={training_mode!r} / model={model_type!r} is not "
                f"score-based. dds/dynamic_dps are posterior samplers for score "
                f"models; cold diffusion and non-diffusion paradigms cannot run them "
                f"(the sampler fails loud at runtime)."
            ),
            "error",
            category="sampler_model_compat",
            yaml_keys=[
                "training.diffusion.inference_sampler",
                "training.training_mode",
                "model.model_type",
            ],
            fix_hint=(
                "Use a score-based diffusion arm (training_mode: diffusion with a "
                "score_based_diffusion model), or pick a sampler compatible with the "
                "paradigm (e.g. cold_mri for kspace_cold_diffusion)."
            ),
        )

    @staticmethod
    def _is_equivariant_imaging_arm(config: "TrainingSettings") -> bool:
        """True iff the arm selects EquivariantImagingStrategy (block or key)."""
        training = getattr(config, "training", None)
        if getattr(training, "equivariant_imaging", None) is not None:
            return True
        strategy_class = str(getattr(training, "strategy_class", "") or "")
        if strategy_class.endswith("equivariant_imaging_strategy.EquivariantImagingStrategy"):
            return True
        training_mode = str(getattr(training, "training_mode", "") or "").lower()
        return training_mode in {"equivariant_imaging", "robust_ei"}

    def check_ei_sensing_margin(self, config: "TrainingSettings") -> HealthCheckResult:
        """Equivariant Imaging identifiability gate (Tachella sensing theorems).

        EI learns the null space only when the forward operator A breaks the
        group symmetry — i.e. there must be undersampling. With full sampling
        A = F is invertible/equivariant to the group, the group keeps signal
        inside the measured subspace, and EI adds nothing (the numeric
        ``sensing_margin`` -> 0). This is the static, config-level guard; the
        Tier-2 ``ei_forward_probe`` computes the margin numerically.
        """
        check_name = "ei_sensing_margin"
        if not self._is_equivariant_imaging_arm(config):
            return HealthCheckResult(
                True,
                check_name,
                "Not an Equivariant Imaging arm — not applicable.",
                "info",
            )
        physics = getattr(config, "physics", None)
        cs = getattr(physics, "compressed_sensing", None)
        accel = getattr(config, "undersampling", None)
        cs_factor = getattr(cs, "acceleration_factor", None)
        base_accel = getattr(accel, "base_acceleration", None)
        rates = [r for r in (cs_factor, base_accel) if isinstance(r, (int, float))]
        undersampled = any(r > 1.0 for r in rates)
        if undersampled:
            return HealthCheckResult(
                True,
                check_name,
                f"EI arm is undersampled (rates={rates}) — operator breaks the "
                "group symmetry, so EI is identifiable.",
                "info",
                category="equivariant_imaging",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                "Equivariant Imaging arm has no undersampling "
                f"(acceleration rates={rates or 'none'} <= 1). With full sampling "
                "the forward operator is equivariant to the group and EI cannot "
                "recover null-space signal (sensing_margin -> 0): the equivariance "
                "term is a no-op."
            ),
            "error",
            category="equivariant_imaging",
            yaml_keys=[
                "physics.compressed_sensing.acceleration_factor",
                "acceleration.base_acceleration",
            ],
            fix_hint=(
                "Undersample the acquisition (acceleration_factor > 1) so the "
                "forward operator breaks the chosen group symmetry."
            ),
        )

    def check_ei_robust_key_matches_correction(
        self, config: "TrainingSettings"
    ) -> HealthCheckResult:
        """The 'robust_ei' key must honour its promise of the noise correction.

        Both ``equivariant_imaging`` and ``robust_ei`` map to the same strategy
        class; the distinguishing signal is ``training_mode``. Selecting
        ``robust_ei`` with ``robust_correction=false`` is a contradiction (a
        registry key that silently does nothing — pitfall #16); reject it.
        """
        check_name = "ei_robust_key_matches_correction"
        training = getattr(config, "training", None)
        training_mode = str(getattr(training, "training_mode", "") or "").lower()
        if training_mode != "robust_ei":
            return HealthCheckResult(
                True,
                check_name,
                "training_mode is not 'robust_ei' — not applicable.",
                "info",
            )
        ei = getattr(training, "equivariant_imaging", None)
        robust = bool(getattr(ei, "robust_correction", False))
        if robust:
            return HealthCheckResult(
                True,
                check_name,
                "robust_ei arm has robust_correction=true (key honoured).",
                "info",
                category="equivariant_imaging",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                "training_mode='robust_ei' but equivariant_imaging.robust_correction "
                "is false/missing — the key advertises the nc-χ GSURE correction "
                "but it is off. Set robust_correction=true (and noise_std_estimate), "
                "or use training_mode='equivariant_imaging' for the plain arm."
            ),
            "error",
            category="equivariant_imaging",
            yaml_keys=[
                "training.training_mode",
                "training.equivariant_imaging.robust_correction",
            ],
            fix_hint=(
                "Set training.equivariant_imaging.robust_correction=true and "
                "noise_std_estimate, or switch to training_mode='equivariant_imaging'."
            ),
        )

    @staticmethod
    def _ssdu_block(config: "TrainingSettings"):
        """Return the training.ssdu sub-block, or None if not an SSDU arm."""
        training = getattr(config, "training", None)
        ssdu = getattr(training, "ssdu", None)
        if ssdu is not None:
            return ssdu
        strategy_class = str(getattr(training, "strategy_class", "") or "")
        training_mode = str(getattr(training, "training_mode", "") or "").lower()
        if strategy_class.endswith("ssdu_strategy.SSDUReconstructionStrategy") or training_mode in {
            "ssdu",
            "self_supervised_reconstruction",
            "robust_ssdu",
        }:
            return ssdu  # may be None — caller treats absent block as defaults
        return False  # sentinel: not an SSDU arm

    def check_ssdu_selection_density_range(self, config: "TrainingSettings") -> HealthCheckResult:
        """Warn if the SSDU Theta selection density is outside the useful range.

        Yaman et al. and the SSDU literature use a held-out fraction roughly in
        [0.2, 0.5]: too small starves the loss of held-out points; too large
        leaks too much of the acquired signal out of the network input, the chief
        SSDU weakness at high acceleration.
        """
        check_name = "ssdu_selection_density_range"
        block = self._ssdu_block(config)
        if block is False:
            return HealthCheckResult(True, check_name, "Not an SSDU arm — not applicable.", "info")
        theta_fraction = float(getattr(block, "theta_fraction", 0.4))
        if 0.2 <= theta_fraction <= 0.5:
            return HealthCheckResult(
                True,
                check_name,
                f"SSDU theta_fraction={theta_fraction:.3g} is in the useful range [0.2, 0.5].",
                "info",
                category="ssdu",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                f"SSDU theta_fraction={theta_fraction:.3g} is outside the useful "
                "range [0.2, 0.5]: <0.2 starves the held-out loss, >0.5 leaks too "
                "much acquired signal out of the network input."
            ),
            "warning",
            category="ssdu",
            yaml_keys=["training.ssdu.theta_fraction"],
            fix_hint="Set training.ssdu.theta_fraction in [0.2, 0.5] (0.4 is typical).",
        )

    def check_robust_ssdu_key_matches_correction(
        self, config: "TrainingSettings"
    ) -> HealthCheckResult:
        """The 'robust_ssdu' key must honour its Noisier2Noise correction.

        Both ``ssdu`` and ``robust_ssdu`` map to the same strategy class; the
        distinguishing signal is ``training_mode``. Selecting ``robust_ssdu`` with
        ``noisier2noise_correction=false`` is a contradiction (a key that silently
        does nothing — pitfall #16).
        """
        check_name = "robust_ssdu_key_matches_correction"
        training = getattr(config, "training", None)
        training_mode = str(getattr(training, "training_mode", "") or "").lower()
        if training_mode != "robust_ssdu":
            return HealthCheckResult(
                True,
                check_name,
                "training_mode is not 'robust_ssdu' — not applicable.",
                "info",
            )
        ssdu = getattr(training, "ssdu", None)
        correction = bool(getattr(ssdu, "noisier2noise_correction", False))
        if correction:
            return HealthCheckResult(
                True,
                check_name,
                "robust_ssdu arm has noisier2noise_correction=true (key honoured).",
                "info",
                category="ssdu",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                "training_mode='robust_ssdu' but ssdu.noisier2noise_correction is "
                "false/missing — the key advertises the Noisier2Noise correction "
                "but it is off. Set noisier2noise_correction=true (and "
                "noise_std_estimate), or use training_mode='ssdu' for vanilla SSDU."
            ),
            "error",
            category="ssdu",
            yaml_keys=[
                "training.training_mode",
                "training.ssdu.noisier2noise_correction",
            ],
            fix_hint=(
                "Set training.ssdu.noisier2noise_correction=true and "
                "noise_std_estimate, or switch to training_mode='ssdu'."
            ),
        )

    def check_ambient_requires_score_model(self, config: "TrainingSettings") -> HealthCheckResult:
        """Ambient Diffusion needs a score/eps model + k-space input.

        AmbientDiffusionStrategy runs the diffusion forward (eps → x₀ via
        ``_predict_start_from_noise``) on the Λ zero-filled measurement, so it
        requires a score/eps-prediction generator and k-space input (the held-out
        Θ consistency is a k-space residual). A non-diffusion model or image-only
        input fails loud at runtime — reject it at config time (pitfalls #15/#16).
        """
        check_name = "ambient_requires_score_model"
        training = getattr(config, "training", None)
        strategy_class = str(getattr(training, "strategy_class", "") or "")
        training_mode = str(getattr(training, "training_mode", "") or "").lower()
        is_ambient = (
            getattr(training, "ambient", None) is not None
            or strategy_class.endswith("ambient_diffusion_strategy.AmbientDiffusionStrategy")
            or training_mode == "ambient_diffusion"
        )
        if not is_ambient:
            return HealthCheckResult(
                True,
                check_name,
                "Not an Ambient Diffusion arm — not applicable.",
                "info",
            )
        model_type = str(getattr(getattr(config, "model", None), "model_type", "") or "").lower()
        input_domain = str(getattr(training, "input_domain", "") or "").lower()
        dataset_type = str(getattr(getattr(config, "data", None), "dataset_type", "") or "").lower()
        is_score = ("score" in model_type or "diffusion" in model_type) and "cold" not in model_type
        has_kspace = "kspace" in input_domain or "kspace" in dataset_type
        if is_score and has_kspace:
            return HealthCheckResult(
                True,
                check_name,
                f"Ambient arm uses a score/diffusion model ({model_type!r}) on k-space input.",
                "info",
                category="ambient_diffusion",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                f"Ambient Diffusion requires a score/eps diffusion model + k-space "
                f"input, but model={model_type!r}, input_domain={input_domain!r}, "
                f"dataset_type={dataset_type!r}. The eps→x₀ path and held-out Θ "
                "k-space consistency cannot run otherwise."
            ),
            "error",
            category="ambient_diffusion",
            yaml_keys=[
                "model.model_type",
                "training.input_domain",
                "data.dataset_type",
            ],
            fix_hint=(
                "Use a score_based_diffusion model with dataset_type=kspace / input_domain=kspace."
            ),
        )

    def check_operator_posterior_requires_complex(
        self, config: "TrainingSettings"
    ) -> HealthCheckResult:
        """T3 operator posterior is only identifiable on complex k-space.

        A calibrated posterior over the BCH operator (``operator_id.operator_posterior:
        laplace``) is meaningful only when the operator is identifiable — which
        requires complex multi-coil k-space (Ahmed–Recht–Romberg bilinear
        identifiability). On single-contrast magnitude data the operator is
        non-identifiable and the posterior is uninformative; reject it at config
        time so the knob is not an over-claim (pitfalls #15/#19).
        """
        check_name = "operator_posterior_requires_complex"
        training = getattr(config, "training", None)
        op = getattr(training, "operator_id", None)
        posterior = str(getattr(op, "operator_posterior", "none") or "none").lower()
        if posterior == "none":
            return HealthCheckResult(
                True,
                check_name,
                "operator_posterior is 'none' — point estimate, not applicable.",
                "info",
            )
        input_domain = str(getattr(training, "input_domain", "") or "").lower()
        dataset_type = str(getattr(getattr(config, "data", None), "dataset_type", "") or "").lower()
        has_kspace = "kspace" in input_domain or "kspace" in dataset_type
        if has_kspace:
            return HealthCheckResult(
                True,
                check_name,
                f"operator_posterior='{posterior}' on complex k-space input — identifiable.",
                "info",
                category="operator_id",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                f"operator_id.operator_posterior='{posterior}' requires complex "
                f"multi-coil k-space (Ahmed–Recht–Romberg identifiability), but "
                f"input_domain={input_domain!r} / dataset_type={dataset_type!r} is "
                "not k-space. On magnitude data the operator is non-identifiable "
                "and the posterior is uninformative (an over-claim)."
            ),
            "error",
            category="operator_id",
            yaml_keys=[
                "training.operator_id.operator_posterior",
                "training.input_domain",
                "data.dataset_type",
            ],
            fix_hint=(
                "Use complex k-space input (dataset_type/input_domain=kspace), or "
                "set operator_id.operator_posterior='none' for the point estimate."
            ),
        )

    def check_trajectory_metric_requires_matching_parametrization(
        self, config: "TrainingSettings"
    ) -> HealthCheckResult:
        """The static half of the trajectory parametrization lock (pitfall #16).

        ``k_space_trajectory_rmse`` grades a spiral ``Δk(t)``. If a config
        requests it, the chosen model MUST declare
        ``trajectory_parametrization == "spiral"`` — otherwise it is the
        diff_trajectory_opt facade (a Cartesian per-readout-line ``Δk`` graded
        against a spiral one). The runtime guard's shape check is the second lock.
        """
        check_name = "trajectory_metric_requires_matching_parametrization"
        validation = getattr(config, "validation", None)
        metrics = (validation.scoring.compute if validation else None) or []
        requested = {m.lower() for m in metrics if isinstance(m, str)}
        if not (requested & {"k_space_trajectory_rmse", "traj_rmse"}):
            return HealthCheckResult(
                True,
                check_name,
                "k_space_trajectory_rmse not requested — parametrization lock N/A.",
                "info",
            )
        model = getattr(config, "model", None)
        model_type = getattr(model, "model_type", None)
        param = None
        try:
            from spectramr.models.registry import get_model_capabilities

            caps = get_model_capabilities(model_type) if model_type else None
            param = getattr(caps, "trajectory_parametrization", None) if caps else None
        except Exception:
            param = None
        if param == "spiral":
            return HealthCheckResult(
                True,
                check_name,
                f"model {model_type!r} declares trajectory_parametrization='spiral'.",
                "info",
                category="trajectory_metric_parametrization",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                f"validation.metrics requests 'k_space_trajectory_rmse' (a spiral "
                f"Δk metric) but model {model_type!r} does not declare "
                f"trajectory_parametrization='spiral' (got {param!r}). Grading a "
                f"Cartesian per-line Δk as a spiral trajectory is a facade "
                f"(pitfall #16)."
            ),
            "error",
            category="trajectory_metric_parametrization",
            yaml_keys=["validation.metrics", "model.model_type"],
            fix_hint=(
                "Use a model that declares trajectory_parametrization='spiral' "
                "(e.g. spiral_trajectory_estimator), or drop k_space_trajectory_rmse."
            ),
        )

    def check_trajectory_recon_requires_measured_emit(
        self, config: "TrainingSettings"
    ) -> HealthCheckResult:
        """``read_measured_trajectory`` requires the dataset to emit the pair.

        If ``training.trajectory_recon.read_measured_trajectory`` is set, the
        dataset must emit ``trajectory_measured``/``trajectory_nominal`` — i.e.
        ``data.ismrmrd.emit_paired_trajectory`` must be true. Otherwise the
        supervised Δk loss has no reference (a silent no-op, pitfall #15).
        """
        check_name = "trajectory_recon_requires_measured_emit"
        training = getattr(config, "training", None)
        tr = getattr(training, "trajectory_recon", None)
        read_measured = bool(getattr(tr, "read_measured_trajectory", False))
        if not read_measured:
            return HealthCheckResult(
                True,
                check_name,
                "trajectory_recon.read_measured_trajectory not set — emit check N/A.",
                "info",
            )
        data = getattr(config, "data", None)
        ismrmrd = getattr(data, "ismrmrd", None)
        emit = bool(getattr(ismrmrd, "emit_paired_trajectory", False))
        if emit:
            return HealthCheckResult(
                True,
                check_name,
                "dataset emits paired nominal+measured trajectories.",
                "info",
                category="trajectory_recon_emit",
            )
        return HealthCheckResult(
            False,
            check_name,
            (
                "training.trajectory_recon.read_measured_trajectory=true but "
                "data.ismrmrd.emit_paired_trajectory is not set — the supervised "
                "Δk loss would have no reference (pitfall #15)."
            ),
            "error",
            category="trajectory_recon_emit",
            yaml_keys=[
                "training.trajectory_recon.read_measured_trajectory",
                "data.ismrmrd.emit_paired_trajectory",
            ],
            fix_hint="Set data.ismrmrd.emit_paired_trajectory: true (+ nominal_file_glob).",
        )

    def check_operator_id_config(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Tier-1 checks for the operator-identification paradigm (Proposal 1).

        Only fires when ``training.operator_id`` is present. Validates that:

        * every ``mode_dictionary`` entry resolves to a generator adapter;
        * ``1 <= covariance_rank < N_pixels``;
        * ``bch_order`` is implemented (1, 2 or 3);
        * ``1 <= krylov_dim <= N_pixels`` (WARNING when large vs the image).
        """
        results: list[HealthCheckResult] = []
        training = getattr(config, "training", None)
        op = getattr(training, "operator_id", None) if training else None
        if op is None:
            return results

        # ``operator_id`` may be a dict (extra="allow" on the training schema)
        # or a typed model — normalise field access either way.
        def _get(name: str, default=None):
            if isinstance(op, dict):
                return op.get(name, default)
            return getattr(op, name, default)

        mode_dictionary = _get("mode_dictionary", ()) or ()
        covariance_rank = int(_get("covariance_rank", 8) or 8)
        bch_order = int(_get("bch_order", 2) or 2)
        krylov_dim = int(_get("krylov_dim", 30) or 30)

        # N_pixels = C * H * W from the model/data geometry.
        try:
            patch = list(config.data.sampling.patch_size or [])
            spatial = [int(s) for s in patch[:2]] if len(patch) >= 2 else [64, 64]
            channels = int(getattr(config.model, "in_channels", 1) or 1)
            n_pixels = max(1, channels * spatial[0] * spatial[1])
        except Exception:
            n_pixels = 64 * 64

        # 1) operator_basis_registered
        try:
            from spectramr.infrastructure.physics.degradation_generators import (
                GENERATOR_ADAPTERS,
            )

            missing = [m for m in mode_dictionary if m not in GENERATOR_ADAPTERS]
        except Exception as exc:  # pragma: no cover — import guard
            missing = []
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="operator_basis_registered",
                    message=f"generator registry unavailable for validation ({exc}).",
                    severity="info",
                    category="operator_id",
                )
            )
        if mode_dictionary:
            results.append(
                HealthCheckResult(
                    passed=not missing,
                    check_name="operator_basis_registered",
                    message=(
                        f"all {len(mode_dictionary)} modes resolve to generator adapters."
                        if not missing
                        else f"unregistered modes: {missing}."
                    ),
                    severity="error" if missing else "info",
                    category="operator_id",
                    yaml_keys=["training.operator_id.mode_dictionary"],
                    fix_hint=(
                        "Register an adapter in "
                        "infrastructure.physics.degradation_generators or remove "
                        "the unknown mode."
                        if missing
                        else None
                    ),
                )
            )

        # 2) covariance_rank_bound
        rank_ok = 1 <= covariance_rank < n_pixels
        results.append(
            HealthCheckResult(
                passed=rank_ok,
                check_name="covariance_rank_bound",
                message=(
                    f"covariance_rank={covariance_rank} ∈ [1, {n_pixels})."
                    if rank_ok
                    else f"covariance_rank={covariance_rank} must satisfy 1 <= r < N_pixels={n_pixels}."
                ),
                severity="info" if rank_ok else "error",
                category="operator_id",
                yaml_keys=["training.operator_id.covariance_rank"],
                fix_hint=None if rank_ok else f"Set 1 <= covariance_rank < {n_pixels}.",
            )
        )

        # 3) bch_order_supported
        order_ok = bch_order in (1, 2, 3)
        results.append(
            HealthCheckResult(
                passed=order_ok,
                check_name="bch_order_supported",
                message=(
                    f"bch_order={bch_order} is implemented."
                    if order_ok
                    else f"bch_order={bch_order} unsupported (only 1, 2, 3 implemented)."
                ),
                severity="info" if order_ok else "error",
                category="operator_id",
                yaml_keys=["training.operator_id.bch_order"],
                fix_hint=None if order_ok else "Set bch_order to 1, 2 or 3.",
            )
        )

        # 4) krylov_dim_valid
        krylov_ok = 1 <= krylov_dim <= n_pixels
        krylov_large = krylov_dim > max(1, n_pixels // 4)
        if not krylov_ok:
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="krylov_dim_valid",
                    message=f"krylov_dim={krylov_dim} must satisfy 1 <= m <= N_pixels={n_pixels}.",
                    severity="error",
                    category="operator_id",
                    yaml_keys=["training.operator_id.krylov_dim"],
                    fix_hint=f"Set 1 <= krylov_dim <= {n_pixels}.",
                )
            )
        elif krylov_large:
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="krylov_dim_valid",
                    message=(
                        f"krylov_dim={krylov_dim} is large relative to N_pixels={n_pixels}; "
                        "the Krylov action will be expensive."
                    ),
                    severity="warning",
                    category="operator_id",
                    yaml_keys=["training.operator_id.krylov_dim"],
                    fix_hint="Reduce krylov_dim; 20-40 is typical for stiff generators.",
                )
            )
        else:
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="krylov_dim_valid",
                    message=f"krylov_dim={krylov_dim} ∈ [1, {n_pixels}].",
                    severity="info",
                    category="operator_id",
                )
            )

        return results

    def check_acq_hypernetwork_config(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Tier-1 checks for LCAH (contrast/field-agnostic bundle M3).

        Only fires when ``training.acq_hypernetwork`` is present. Validates:

        * ``acq_vector_present`` (ERROR) — the data pipeline actually emits the
          acquisition vector the hypernetwork is conditioned on. Without it the
          run trains a constant FiLM and the certificate means nothing;
        * ``spectral_norm_enabled`` (WARNING) — the Lipschitz certificate is
          *void* without spectral normalisation, so an ablation is allowed but
          must be visible.
        """
        results: list[HealthCheckResult] = []
        training = getattr(config, "training", None)
        acq = getattr(training, "acq_hypernetwork", None) if training else None
        if acq is None:
            return results

        def _get(name: str, default=None):
            if isinstance(acq, dict):
                return acq.get(name, default)
            return getattr(acq, name, default)

        acquisition_key = str(_get("acquisition_key", "acquisition") or "acquisition")
        # The per-sample (TE, TR, TI, FA, B0) seam is
        # ``data.acquisition_metadata`` — the same block AcquisitionEmbedding and
        # BlochConsistencyLoss consume. `multi_contrast.acquisition_params`
        # carries fixed per-CLASS rows instead, which is enough to condition an
        # arm whose protocol is constant per contrast.
        data = getattr(config, "data", None)
        acq_meta = getattr(data, "acquisition_metadata", None) if data else None
        per_sample = bool(getattr(acq_meta, "enabled", False))
        multi_contrast = getattr(data, "multi_contrast", None) if data else None
        per_class = bool(getattr(multi_contrast, "acquisition_params", None))
        exposes = per_sample or per_class
        route = (
            "data.acquisition_metadata.enabled"
            if per_sample
            else ("data.multi_contrast.acquisition_params" if per_class else "none")
        )
        results.append(
            HealthCheckResult(
                passed=exposes,
                check_name="acq_vector_present",
                message=(
                    f"Acquisition vector route: {route}; LCAH conditions on "
                    f"batch['{acquisition_key}']."
                ),
                severity="info" if exposes else "error",
                category="acq_hypernetwork",
                yaml_keys=[
                    "data.acquisition_metadata.enabled",
                    "data.multi_contrast.acquisition_params",
                    "training.acq_hypernetwork.acquisition_key",
                ],
                fix_hint=(
                    None
                    if exposes
                    else (
                        "Set data.acquisition_metadata.enabled: true (per-sample "
                        "TE/TR/TI/FA/B0), or declare "
                        "data.multi_contrast.acquisition_params for a "
                        "fixed-per-contrast protocol. Without either, the "
                        "hypernetwork trains a constant FiLM."
                    )
                ),
            )
        )

        spectral_norm = bool(_get("spectral_norm", True))
        results.append(
            HealthCheckResult(
                passed=spectral_norm,
                check_name="spectral_norm_enabled",
                message=(
                    "Spectral norm on; Lipschitz certificate valid."
                    if spectral_norm
                    else "spectral_norm=false — the LCAH certificate is VOID for this arm."
                ),
                severity="info" if spectral_norm else "warning",
                category="acq_hypernetwork",
                yaml_keys=["training.acq_hypernetwork.spectral_norm"],
                fix_hint=(
                    None
                    if spectral_norm
                    else (
                        "Set training.acq_hypernetwork.spectral_norm: true, or keep "
                        "it off deliberately and do not report a certified radius."
                    )
                ),
            )
        )
        return results

    def check_mcgi_invariance_declared(self, config: TrainingSettings) -> HealthCheckResult:
        """Tier-1 for MCGI (bundle M2): the rank transform must be in the pipeline.

        ``mcgi_encoder``'s invariance to monotone intensity remaps is a property
        of the rank transform :math:`R` sitting in front of the backbone. An arm
        that selects the encoder but disables rank normalisation gets an ordinary
        conv net while still *claiming* exact :math:`G_+\\rtimes\\mathbb Z_2`
        invariance — a facade mechanism (pitfall #16), so this is an ERROR.
        """
        name = "mcgi_invariance_declared"
        model_name = str(getattr(getattr(config, "model", None), "name", "") or "")
        if model_name != "mcgi_encoder":
            return HealthCheckResult(
                passed=True,
                check_name=name,
                message="Arm does not select mcgi_encoder; check skipped.",
                severity="info",
                category="mcgi",
            )
        kwargs = getattr(getattr(config, "model", None), "model_kwargs", None) or {}
        if not isinstance(kwargs, dict):
            kwargs = dict(getattr(kwargs, "__dict__", {}) or {})
        # The guarantee needs the exact (hard) rank at inference; a soft rank is
        # a training-time relaxation only.
        hard_rank = bool(kwargs.get("hard_rank_eval", True))
        return HealthCheckResult(
            passed=hard_rank,
            check_name=name,
            message=(
                "mcgi_encoder uses the exact hard rank at inference; G_+ invariance holds exactly."
                if hard_rank
                else "mcgi_encoder has hard_rank_eval=false — invariance is only "
                "approximate, so the exactness claim does not hold."
            ),
            severity="info" if hard_rank else "error",
            category="mcgi",
            yaml_keys=["model.model_kwargs.hard_rank_eval"],
            fix_hint=(
                None
                if hard_rank
                else (
                    "Set model.model_kwargs.hard_rank_eval: true, or stop "
                    "describing this arm as exactly contrast-invariant."
                )
            ),
        )

    def check_dispersion_bloch_ae_config(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Tier-1 checks for DL-BAE (contrast/field-agnostic bundle M4).

        Only fires when ``training.dispersion_bloch_ae`` is present. Validates:

        * ``dispersion_identifiability`` (ERROR) — a ``P``-pool BPP model has
          ``2P+1`` free constants per rate, so the arm needs ``M >= 2P+1``
          *distinct* fields. Below that the fit is rank-deficient and the
          recovered latent is meaningless, which no amount of training fixes;
        * ``dispersion_monotonicity_weight_positive`` (WARNING) — with the
          hinge disabled the fit may land on a physically impossible
          ``dT1/dB0 < 0``, voiding cross-field extrapolation.
        """
        results: list[HealthCheckResult] = []
        training = getattr(config, "training", None)
        dl = getattr(training, "dispersion_bloch_ae", None) if training else None
        if dl is None:
            return results

        def _get(name: str, default=None):
            if isinstance(dl, dict):
                return dl.get(name, default)
            return getattr(dl, name, default)

        n_pools = int(_get("n_pools", 1) or 1)
        fields = tuple(_get("fields_present", ()) or ())
        distinct = len({float(b) for b in fields})
        required = 2 * n_pools + 1
        identifiable = distinct >= required
        results.append(
            HealthCheckResult(
                passed=identifiable,
                check_name="dispersion_identifiability",
                message=(
                    f"n_pools={n_pools} needs M >= {required} distinct fields; "
                    f"fields_present supplies {distinct}."
                ),
                severity="info" if identifiable else "error",
                category="dispersion_bloch_ae",
                yaml_keys=[
                    "training.dispersion_bloch_ae.n_pools",
                    "training.dispersion_bloch_ae.fields_present",
                ],
                fix_hint=(
                    None
                    if identifiable
                    else (
                        f"Reduce n_pools to {max(1, (distinct - 1) // 2)} or add "
                        f"fields so fields_present names >= {required} distinct B0 values."
                    )
                ),
            )
        )

        mono_weight = float(_get("monotonicity_weight", 1.0) or 0.0)
        mono_on = mono_weight > 0.0
        results.append(
            HealthCheckResult(
                passed=mono_on,
                check_name="dispersion_monotonicity_weight_positive",
                message=(
                    f"monotonicity_weight={mono_weight} enforces dT1/dB0 >= 0."
                    if mono_on
                    else "monotonicity_weight=0 — dT1/dB0 >= 0 is NOT enforced."
                ),
                severity="info" if mono_on else "warning",
                category="dispersion_bloch_ae",
                yaml_keys=["training.dispersion_bloch_ae.monotonicity_weight"],
                fix_hint=(
                    None
                    if mono_on
                    else (
                        "Set training.dispersion_bloch_ae.monotonicity_weight > 0, or "
                        "keep it at 0 deliberately and do not claim cross-field transfer."
                    )
                ),
            )
        )
        return results

    def check_field_cocycle_arm(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Tier-1 guards for the cocycle-consistent unified operator (idea 4.2).

        Enforces the three preconditions that keep the arm scientifically coherent:
        (1) a non-trivial fidelity/adversarial term so identity+cocycle cannot
        collapse to ``G=Id`` (pitfall #20); (2) the reference field lies in range;
        (3) the generator is the registered *unified* single model (no per-field
        routing) — the static form of the Task-3 anti-ensemble requirement.
        """
        results: list[HealthCheckResult] = []
        training = getattr(config, "training", None)
        mode = (getattr(training, "training_mode", "") or "") if training else ""
        strat = (getattr(training, "strategy_class", "") or "") if training else ""
        if "field_cocycle" not in (mode, strat):
            return [
                HealthCheckResult(
                    True,
                    "field_cocycle_arm",
                    "not a field_cocycle arm.",
                    "info",
                )
            ]
        cfg = getattr(training, "field_cocycle", None)

        def _w(name: str, default: float) -> float:
            return float(getattr(cfg, name, default)) if cfg is not None else default

        cocycle = _w("cocycle_weight", 1.0)
        identity = _w("identity_weight", 0.5)
        adversarial = _w("adversarial_weight", 1.0)
        # (1) fidelity/adversarial must carry weight — paired L1 is always on in the
        # strategy, so the operative additional guard is adversarial_weight > 0.
        if (cocycle + identity) > 0 and adversarial <= 0:
            results.append(
                HealthCheckResult(
                    False,
                    "field_cocycle_fidelity_nonzero",
                    "cocycle/identity terms are active but adversarial_weight=0; "
                    "identity+cocycle alone admit the trivial G=Id solution.",
                    "error",
                    category="degenerate_solution",
                    yaml_keys=["training.field_cocycle.adversarial_weight"],
                    fix_hint="set training.field_cocycle.adversarial_weight > 0 (the "
                    "paired L1 is always on; the adversarial term breaks the tie).",
                )
            )
        else:
            results.append(
                HealthCheckResult(
                    True,
                    "field_cocycle_fidelity_nonzero",
                    "a non-trivial fidelity/adversarial term is present.",
                    "info",
                )
            )
        # (2) reference field in range (config validator also enforces; belt+braces).
        ref = _w("reference_field_tesla", 3.0)
        fmin = _w("field_min_tesla", 0.1)
        fmax = _w("field_max_tesla", 7.0)
        if not (fmin <= ref <= fmax):
            results.append(
                HealthCheckResult(
                    False,
                    "field_cocycle_reference_in_range",
                    f"reference_field_tesla={ref} is outside [{fmin}, {fmax}].",
                    "error",
                    category="config",
                    yaml_keys=["training.field_cocycle.reference_field_tesla"],
                    fix_hint="the trivialising reference field must lie within the log-field axis.",
                )
            )
        else:
            results.append(
                HealthCheckResult(
                    True,
                    "field_cocycle_reference_in_range",
                    f"reference_field_tesla={ref} ∈ [{fmin}, {fmax}].",
                    "info",
                )
            )
        # (3) single-model / anti-ensemble: the generator must be a registered model
        # advertising is_unified_single_model=True.
        model_type = getattr(getattr(config, "model", None), "model_type", None)
        unified = False
        try:
            from spectramr.models.registry import get_model_class

            unified = bool(
                getattr(get_model_class(str(model_type)), "is_unified_single_model", False)
            )
        except Exception:
            # Registry not populated in this context: fall back to the known unified
            # generator name so the guard still fires on an obviously-wrong model.
            unified = model_type == "field_cocycle_generator"
        if not unified:
            results.append(
                HealthCheckResult(
                    False,
                    "field_cocycle_single_model",
                    f"model_type={model_type!r} does not advertise "
                    "is_unified_single_model=True; Task 3 forbids per-field routing / "
                    "functional ensembles.",
                    "error",
                    category="facade",
                    yaml_keys=["model.model_type"],
                    fix_hint="use the unified field_cocycle_generator (one conditioned "
                    "generator realises every ordered field pair).",
                )
            )
        else:
            results.append(
                HealthCheckResult(
                    True,
                    "field_cocycle_single_model",
                    f"model_type={model_type!r} is a unified single model.",
                    "info",
                )
            )
        return results

    def check_bloch_synth_arm(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Tier-1 guards for the relaxometry + Bloch-synthesis arm (idea 2.1).

        (1) identifiability: at least 3 source contrasts AND ``model.in_channels``
        matching the stacked count (Proposition 3, J>=3); (2) the dispersion envelope
        is physiological. Target-field range and the ``segmenter='none' => seg=0`` rule
        are enforced at config-load by ``BlochSynthConfig``.
        """
        results: list[HealthCheckResult] = []
        training = getattr(config, "training", None)
        mode = (getattr(training, "training_mode", "") or "") if training else ""
        strat = (getattr(training, "strategy_class", "") or "") if training else ""
        if "bloch_synth" not in (mode, strat):
            return [HealthCheckResult(True, "bloch_synth_arm", "not a bloch_synth arm.", "info")]
        cfg = getattr(training, "bloch_synth", None)
        contrasts = list(getattr(cfg, "source_contrasts", []) or []) if cfg else []
        in_ch = getattr(getattr(config, "model", None), "in_channels", None)
        n = len(contrasts)
        if n < 3:
            results.append(
                HealthCheckResult(
                    False,
                    "bloch_synth_source_contrast_count",
                    f"only {n} source contrast(s) declared; the relaxometry inversion "
                    "is under-determined (Proposition 3 needs J>=3).",
                    "error",
                    category="hypothesis_untestable",
                    yaml_keys=["training.bloch_synth.source_contrasts"],
                    fix_hint="declare at least 3 source contrasts (e.g. "
                    "[T1w, T2w, FLAIR]) and set model.in_channels to match.",
                )
            )
        elif in_ch is not None and int(in_ch) != n:
            results.append(
                HealthCheckResult(
                    False,
                    "bloch_synth_source_contrast_count",
                    f"model.in_channels={in_ch} != len(source_contrasts)={n}; the "
                    "encoder input would not match the stacked contrast tensor.",
                    "error",
                    category="config",
                    yaml_keys=[
                        "model.in_channels",
                        "training.bloch_synth.source_contrasts",
                    ],
                    fix_hint=f"set model.in_channels: {n}.",
                )
            )
        else:
            results.append(
                HealthCheckResult(
                    True,
                    "bloch_synth_source_contrast_count",
                    f"{n} source contrasts, model.in_channels={in_ch} (J>=3 identifiable).",
                    "info",
                )
            )
        # dispersion envelope sanity (physiological ~ [0.2, 0.5]).
        bounds = getattr(cfg, "dispersion_beta_bounds", (0.3, 0.4)) if cfg else (0.3, 0.4)
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo < 0.2 or hi > 0.5:
            results.append(
                HealthCheckResult(
                    False,
                    "bloch_synth_dispersion_bounds",
                    f"dispersion_beta_bounds=({lo}, {hi}) exceed the physiological "
                    "envelope ~[0.2, 0.5] (Rooney et al.).",
                    "warning",
                    category="physics",
                    yaml_keys=["training.bloch_synth.dispersion_beta_bounds"],
                    fix_hint="keep beta bounds within the measured field-dispersion range.",
                )
            )
        else:
            results.append(
                HealthCheckResult(
                    True,
                    "bloch_synth_dispersion_bounds",
                    f"dispersion_beta_bounds=({lo}, {hi}) ∈ physiological envelope.",
                    "info",
                )
            )
        # seg-consistency ramp with no segmenter is an inert facade (#16): the static
        # seg=0 rule is config-load-enforced, but a loss_schedule ramp on it is not.
        backend = str(getattr(cfg, "segmenter_backend", "label_dice") or "label_dice")
        sched = getattr(config, "loss_schedule", None)
        if backend == "none" and sched is not None and getattr(sched, "enabled", False):
            seg_targets = {
                getattr(r, "target", None) for r in getattr(sched, "rules", []) or []
            } & {"seg_consistency", "segmentation_dice"}
            if seg_targets:
                results.append(
                    HealthCheckResult(
                        False,
                        "bloch_synth_seg_ramp_requires_segmenter",
                        f"loss_schedule ramps {sorted(seg_targets)} but "
                        "segmenter_backend='none', so the seg-consistency term is a "
                        "no-op (returns 0) — an advertised-but-inert curriculum.",
                        "error",
                        category="facade",
                        yaml_keys=[
                            "training.bloch_synth.segmenter_backend",
                            "loss_schedule.rules",
                        ],
                        fix_hint="set segmenter_backend: label_dice (or drop the seg ramp).",
                    )
                )
        return results

    def check_curriculum_targets_consumed(self, config: TrainingSettings) -> HealthCheckResult:
        """A ``loss_schedule`` target must be consumed by something, else the ramp is a
        silent no-op (pitfall #16). For the MRIxFields2026 leads the consumed set is
        known (the strategy's inline terms + the declarative image losses); a target
        outside it is almost certainly a typo that would never fire.
        """
        check_name = "curriculum_targets_consumed"
        training = getattr(config, "training", None)
        mode = (getattr(training, "training_mode", "") or "") if training else ""
        strat = (getattr(training, "strategy_class", "") or "") if training else ""
        keys = {mode, strat}
        sched = getattr(config, "loss_schedule", None)
        if sched is None or not getattr(sched, "enabled", False):
            return HealthCheckResult(True, check_name, "no loss_schedule configured.", "info")
        # Only scope-check the arms whose consumed target set we know precisely.
        inline: set[str] = set()
        if "field_cocycle" in keys:
            inline = {
                "adversarial",
                "cocycle_consistency",
                "field_identity",
                "latent_cycle",
            }
        elif "bloch_synth" in keys:
            # segmentation_dice is the alias the strategy also consumes
            # (sched.get("seg_consistency", sched.get("segmentation_dice", ...))).
            inline = {
                "seg_consistency",
                "segmentation_dice",
                "bloch_source_consistency",
                "dispersion_prior",
            }
        else:
            return HealthCheckResult(
                True, check_name, f"target-consumption not scoped for {mode!r}.", "info"
            )
        declared: set[str] = set()
        losses_cfg = getattr(config, "losses", None)
        if losses_cfg is not None:
            for attr in LOSS_LIST_DOMAINS:
                for entry in getattr(losses_cfg, attr, None) or []:
                    name = getattr(entry, "name", None) or (
                        entry.get("name") if isinstance(entry, dict) else None
                    )
                    if name:
                        declared.add(str(name))
        consumable = inline | declared
        targets = {getattr(r, "target", None) for r in getattr(sched, "rules", []) or []}
        orphans = sorted(t for t in targets if t and t not in consumable)
        if orphans:
            return HealthCheckResult(
                False,
                check_name,
                f"loss_schedule targets {orphans} are consumed by nothing "
                f"(strategy inline terms {sorted(inline)} or declared image losses "
                f"{sorted(declared)}); the ramp would silently no-op.",
                "warning",
                category="facade",
                yaml_keys=["loss_schedule.rules"],
                fix_hint="target a real inline term or a declared losses.image_losses "
                "entry (nonzero base weight).",
            )
        return HealthCheckResult(
            True,
            check_name,
            f"all {len(targets)} loss_schedule target(s) are consumed.",
            "info",
        )

    def check_curriculum_targets_resolvable(self, config: TrainingSettings) -> HealthCheckResult:
        """Every ``loss_schedule`` target must have a base weight the controller can
        resolve, or the run dies AT THE TRIGGER ITERATION — hours in, on a config that
        passed audit.

        ``LossScheduleController._apply_action`` reads ``self._base_weight(target)``
        unconditionally, and that goes through the loss-weight SSOT
        (:func:`~spectramr.models.losses.weights.build_loss_weight_table`), which sees
        only the ``losses.*`` block. A term whose weight lives solely in a strategy
        block (``training.field_cocycle.cocycle_weight``,
        ``training.bloch_synth.seg_consistency_weight``, ...) is invisible there, so
        ``resolve_loss_weight`` correctly refuses to invent one and raises
        ``ConfigurationError`` (CLAUDE.md #9/#13b).

        This is deliberately NOT the same question as
        :meth:`check_curriculum_targets_consumed`, which asks whether anything READS
        the ramp (facade lens, pitfall #16). ``field_cocycle_anyfield`` passed that
        check — ``cocycle_consistency`` is genuinely consumed by the strategy — and
        still died 22 minutes into the 2026-07-23 sweep because the *base* was
        unresolvable. Consumed and resolvable are independent properties; both need a
        guard.
        """
        check_name = "curriculum_targets_resolvable"
        sched = getattr(config, "loss_schedule", None)
        if sched is None or not getattr(sched, "enabled", False):
            return HealthCheckResult(True, check_name, "no loss_schedule configured.", "info")
        from spectramr.domain.exceptions import ConfigurationError
        from spectramr.models.losses.weights import build_loss_weight_table

        table = build_loss_weight_table(getattr(config, "losses", None))
        unresolvable: list[str] = []
        for rule in getattr(sched, "rules", []) or []:
            target = getattr(rule, "target", None)
            if not target:
                continue
            try:
                # Same call the controller makes (iteration past any warm-up gate).
                table.weight(target, iteration=1_000_000)
            except ConfigurationError:
                unresolvable.append(f"{getattr(rule, 'name', '?')}->{target}")
        if unresolvable:
            return HealthCheckResult(
                False,
                check_name,
                f"loss_schedule rule(s) {sorted(unresolvable)} target a term whose "
                "base weight is declared nowhere in `losses.*`; "
                "LossScheduleController raises ConfigurationError the moment the rule "
                "fires (mid-run, after the trigger iteration).",
                "error",
                category="configuration",
                yaml_keys=["loss_schedule.rules", "losses.image_losses"],
                fix_hint="declare each target on the loss-weight SSOT, e.g. add "
                "`- {name: <target>, weight: <base>}` to `losses.image_losses`. If "
                "the strategy computes the term inline, also list it in that "
                "strategy's `inline_losses` declaration so the fold does not "
                "double-count it.",
            )
        return HealthCheckResult(
            True,
            check_name,
            f"all {len(getattr(sched, 'rules', []) or [])} loss_schedule target(s) "
            "resolve a base weight.",
            "info",
        )

    def check_metric_backend_available(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Fail fast when a requested validation metric has no importable backend.

        torchmetrics-backed metrics (ms_ssim / uqi / kid / fid) report ``NaN`` for the
        WHOLE run when torchmetrics cannot be imported (undeclared dep, or the
        ``huggingface-hub>=1.0`` conflict). LPIPS falls back to the ``lpips`` package, so
        it is only unbacked when BOTH are absent. Catching this at config-health time
        turns a 150k-iter run that silently NaNs a metric — and, if that metric drives
        early stopping / primary_metric, corrupts checkpoint selection — into a loud
        startup error (pitfalls #10/#18). The remedy is an env re-sync on the cluster;
        the check does not fabricate a value.
        """
        from spectramr.core.metrics import evaluation_metrics as em

        check_name = "metric_backend_available"
        val = getattr(config, "validation", None)
        raw = list(val.scoring.compute or []) if val is not None else []
        names = {str(m).lower() for m in raw}
        if not names or em.TORCHMETRICS_AVAILABLE:
            return [HealthCheckResult(True, check_name, "metric backends available.", "info")]

        # torchmetrics missing: these torchmetrics-only metrics have no fallback.
        no_fallback = {"ms_ssim", "msssim", "uqi", "kid", "fid"}
        unbacked = sorted(names & no_fallback)
        if "lpips" in names:
            try:
                import lpips  # noqa: F401
            except ImportError:
                unbacked.append("lpips")
        if not unbacked:
            return [
                HealthCheckResult(
                    True,
                    check_name,
                    "requested metric backends available (lpips fallback ok).",
                    "info",
                )
            ]
        return [
            HealthCheckResult(
                passed=False,
                check_name=check_name,
                message=(
                    f"validation.metrics requests {unbacked} but torchmetrics is not "
                    f"importable ({em._TORCHMETRICS_IMPORT_ERROR!r}); these would report "
                    "NaN for the entire run instead of a real score."
                ),
                severity="error",
                category="metric_backend",
                yaml_keys=["validation.metrics"],
                fix_hint=(
                    "Re-sync the env: `pip install -e '.[dev]'` (torchmetrics needs "
                    "huggingface-hub<1.0). Or drop the unbacked metric(s) from "
                    "validation.scoring."
                ),
            )
        ]

    @staticmethod
    def _model_kwarg(config: Any, key: str, default: Any = None) -> Any:
        """Read ``model.model_kwargs[key]`` tolerating dict- or object-form kwargs."""
        model = getattr(config, "model", None)
        mk = getattr(model, "model_kwargs", None) if model is not None else None
        if mk is None:
            return default
        if isinstance(mk, dict):
            return mk.get(key, default)
        return getattr(mk, key, default)

    def check_contrast_conditioning_strategy_threaded(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """A contrast-widened model requires a strategy that threads ``contrast_id``.

        Anchor: MRIxFields2026 multicontrast rollout, 2026-07-07. The dataset
        always emits ``batch['contrast_id']``; a model widened with
        ``model_kwargs.use_contrast_conditioning: true`` guards on that key
        (#15 — no silent mode-averaging) and RAISES "batch carries no
        'contrast_id'" at step 0 if its ``training.strategy_class`` drops the key
        before the model forward. The allow-list of threaded strategies is the
        SSOT in ``config.validation_constants.CONTRAST_THREADED_STRATEGIES``; this
        moves that step-0 crash to Tier-1 pre-flight.
        """
        enabled = self._model_kwarg(config, "use_contrast_conditioning", False)
        if not enabled:
            return HealthCheckResult(
                passed=True,
                check_name="contrast_conditioning_strategy_threaded",
                message="use_contrast_conditioning off; skipping.",
                severity="info",
            )
        training = getattr(config, "training", None)
        strat = getattr(training, "strategy_class", None) if training else None
        try:
            from spectramr.config.validation_constants import (
                CONTRAST_THREADED_STRATEGIES,
            )
        except Exception as exc:  # pragma: no cover - import safety
            return HealthCheckResult(
                passed=True,
                check_name="contrast_conditioning_strategy_threaded",
                message=f"validation_constants unavailable; skipping ({exc}).",
                severity="info",
            )
        if strat in CONTRAST_THREADED_STRATEGIES:
            return HealthCheckResult(
                passed=True,
                check_name="contrast_conditioning_strategy_threaded",
                message=f"strategy_class={strat!r} threads contrast_id.",
                severity="info",
            )
        return HealthCheckResult(
            passed=False,
            check_name="contrast_conditioning_strategy_threaded",
            message=(
                f"model_kwargs.use_contrast_conditioning=True but "
                f"training.strategy_class={strat!r} does NOT thread contrast_id to "
                f"the model forward. The widened model's #15 guard will raise "
                f"'batch carries no contrast_id' at step 0."
            ),
            severity="error",
            category="silent_fallback",
            yaml_keys=[
                "model.model_kwargs.use_contrast_conditioning",
                "training.strategy_class",
            ],
            fix_hint=(
                "Thread contrast_id through this strategy (mirror "
                "field_flow_strategy: pass contrast_id=batch.get('contrast_id') to "
                "the model in both training and validation) and add it to "
                "CONTRAST_THREADED_STRATEGIES, or set use_contrast_conditioning: "
                f"false. Threaded strategies: {sorted(CONTRAST_THREADED_STRATEGIES)}."
            ),
        )

    def check_mrixfields_pairing_viable(self, config: TrainingSettings) -> list[HealthCheckResult]:
        """Reject an mrixfields config that provably forms 0 pairs at pre-flight.

        Anchor: 2026-07-07 cluster failure — field-pinned pairing policies
        ("fixed_target"/"ulf_source"/"multi_source"/"multi_contrast") produced 0
        pairs because the gitignored val manifest was stale (built before
        ordinal pairing, so every group was a singleton). Two static signals:

        * config-level — a field-pinned policy with no ``mrixfields_target_field``
          (the dataset __init__ raises; the schema does not enforce it);
        * manifest-level — when the manifest IS present, a lightweight
          necessary-condition predictor
          (``mrixfields_pairing_feasibility``) that flags the guaranteed-empty
          case (all-singleton groups / pinned field absent) and names the
          regeneration script. Manifest absent (gitignored) → skip.
        """
        results: list[HealthCheckResult] = []
        data = getattr(config, "data", None)
        dataset_type = getattr(data, "dataset_type", None) if data else None
        if dataset_type != "mrixfields":
            return results

        policy = getattr(getattr(data, "mrixfields", None), "pairing_policy", "all_pairs")
        target_field = getattr(getattr(data, "mrixfields", None), "target_field", None)

        if policy in ("fixed_target", "multi_source", "multi_contrast") and (target_field is None):
            results.append(
                HealthCheckResult(
                    passed=False,
                    check_name="mrixfields_pairing_viable",
                    message=(
                        f"mrixfields_pairing_policy={policy!r} requires "
                        f"data.mrixfields_target_field (Tesla) but it is unset."
                    ),
                    severity="error",
                    category="data_misconfiguration",
                    yaml_keys=[
                        "data.mrixfields_pairing_policy",
                        "data.mrixfields_target_field",
                    ],
                    fix_hint="Set data.mrixfields_target_field (e.g. 7.0).",
                )
            )
            return results  # can't run the manifest predictor without the pin

        try:
            from spectramr.data.datasets.mrixfields_dataset import (
                mrixfields_pairing_feasibility,
            )
            from spectramr.data.split_leakage import read_manifest_records
        except Exception as exc:  # pragma: no cover - import safety
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="mrixfields_pairing_viable",
                    message=f"mrixfields helpers unavailable; skipping ({exc}).",
                    severity="info",
                )
            )
            return results

        checked_any = False
        for label, attr in (
            ("validation", "validation_index_path"),
            ("train", "index_path"),
        ):
            path = getattr(data, attr, None)
            if not path:
                continue
            records = read_manifest_records(self._resolve_manifest_path(str(path)))
            if records is None:
                continue  # gitignored / absent locally → skip this manifest
            checked_any = True
            feasible, reason = mrixfields_pairing_feasibility(
                records, policy=policy, target_field=target_field
            )
            if not feasible:
                results.append(
                    HealthCheckResult(
                        passed=False,
                        check_name="mrixfields_pairing_viable",
                        message=(
                            f"mrixfields_pairing_policy={policy!r} forms 0 pairs on "
                            f"the {label} manifest ({attr}): {reason}"
                        ),
                        severity="error",
                        category="data_misconfiguration",
                        yaml_keys=[
                            f"data.{attr}",
                            "data.mrixfields_pairing_policy",
                        ],
                        fix_hint=(
                            "Regenerate the manifest with "
                            "`python scripts/data/build_mrixfields2026_manifest.py` "
                            "or point the arm at a corpus that carries the pinned "
                            "field(s) in a shared pairing group."
                        ),
                    )
                )
        if checked_any and not any(not r.passed for r in results):
            results.append(
                HealthCheckResult(
                    passed=True,
                    check_name="mrixfields_pairing_viable",
                    message=f"pairing_policy={policy!r} can form pairs on the manifest(s).",
                    severity="info",
                )
            )
        return results

    @staticmethod
    def _resolve_manifest_path(path: str) -> str:
        try:
            from spectramr.data.metadata.path_resolver import PathResolver

            return str(PathResolver.resolve(path))
        except Exception:
            return path

    def check_persistent_workers_on_short_epoch(
        self, config: TrainingSettings
    ) -> HealthCheckResult:
        """Reject ``num_workers>0`` + ``persistent_workers=False`` on a tiny manifest.

        Anchor: 2026-07-10 mrixfields cluster burn. A DataLoader with
        ``persistent_workers=False`` tears down and re-forks its workers at every
        epoch boundary, and each fresh worker re-imports torch + the model registry
        (the ~30-60 s cost the CLI banner warns about). That amortizes fine over a
        long epoch and is ruinous over a short one: 45 volumes at ``batch_size: 4``
        is a 12-iteration epoch, so 4 workers respawned every 12 steps held the GPU
        at ~0% util across ~70 arms for days. It never crashed — the run just
        crawled at ~3.5 s/it, so no crash/dead-loss detector saw it.

        Scoped to dataset types where ONE manifest record yields ONE sample, so
        ``len(records) / batch_size`` is a sound epoch-length bound. Slice-expanding
        types are skipped rather than guessed at (a wrong estimate here would be a
        false positive that blocks a legitimate config).
        """
        data = getattr(config, "data", None)
        dataset_type = str(getattr(data, "dataset_type", "") or "") if data else ""
        if dataset_type not in _ONE_SAMPLE_PER_RECORD_DATASETS:
            return HealthCheckResult(
                passed=True,
                check_name="persistent_workers_on_short_epoch",
                message=(
                    f"dataset_type={dataset_type!r} is not 1-sample-per-record; "
                    f"epoch length is not statically bounded here. n/a."
                ),
                severity="info",
            )

        num_workers = int(getattr(getattr(data, "loader", None), "num_workers", 0) or 0)
        persistent = bool(getattr(getattr(data, "loader", None), "persistent_workers", False))
        if num_workers == 0 or persistent:
            return HealthCheckResult(
                passed=True,
                check_name="persistent_workers_on_short_epoch",
                message=(
                    f"num_workers={num_workers}, persistent_workers={persistent}: "
                    f"no per-epoch worker respawn."
                ),
                severity="info",
            )

        from spectramr.data.split_leakage import read_manifest_records

        records = None
        for attr in ("index_path", "paired_manifest_path", "manifest_path"):
            path = getattr(data, attr, None)
            if not path:
                continue
            records = read_manifest_records(self._resolve_manifest_path(str(path)))
            if records is not None:
                break

        if records is None:
            return HealthCheckResult(
                passed=True,
                check_name="persistent_workers_on_short_epoch",
                message="Train manifest absent (gitignored?); cannot bound epoch length. Skipped.",
                severity="info",
            )

        batch_size = max(1, int(getattr(getattr(data, "loader", None), "batch_size", 1) or 1))
        iters_per_epoch = math.ceil(len(records) / batch_size)
        if iters_per_epoch >= _SHORT_EPOCH_ITERS:
            return HealthCheckResult(
                passed=True,
                check_name="persistent_workers_on_short_epoch",
                message=(f"~{iters_per_epoch} iterations/epoch: worker respawn amortizes."),
                severity="info",
            )

        return HealthCheckResult(
            passed=False,
            check_name="persistent_workers_on_short_epoch",
            message=(
                f"{len(records)} records at batch_size={batch_size} is a "
                f"~{iters_per_epoch}-iteration epoch, but num_workers={num_workers} "
                f"with persistent_workers=False re-forks (and re-imports torch in) "
                f"every worker every {iters_per_epoch} steps. This starves the GPU to "
                f"~0% util without ever crashing."
            ),
            severity="warning",
            category="data_misconfiguration",
            yaml_keys=[
                "data.persistent_workers",
                "data.num_workers",
                "data.batch_size",
            ],
            fix_hint="Set data.persistent_workers: true (keeps workers, and their caches, alive across epochs).",
        )

    def check_train_val_split_leakage(self, config: TrainingSettings) -> HealthCheckResult:
        """Reject a config whose train and validation splits share subjects/files.

        Anchor: data-leak prevention, 2026-07-07. Delegates the record scan to
        the data-layer ``analyze_split_leakage`` (SSOT: file→records stays under
        ``src/spectramr/data/``). Supports every split strategy — two manifests
        (compare directly), single-index ``random`` (a subject straddling the
        deterministic ``split_index`` boundary), ``loso`` (a subject at both the
        holdout site and elsewhere), and the directory crawl (replayed for both
        splits when the data root is on this host; the old "folder-disjoint by
        construction" skip hid the file-level split that leaked subjects across
        the boundary). Manifests / data absent locally → skip (runs on cluster).
        """
        try:
            from spectramr.data.split_leakage import analyze_split_leakage
        except Exception as exc:  # pragma: no cover - import safety
            return HealthCheckResult(
                passed=True,
                check_name="train_val_split_leakage",
                message=f"split-leakage analyzer unavailable; skipping ({exc}).",
                severity="info",
            )
        rep = analyze_split_leakage(config)
        if rep.status == "leak":
            sample = ", ".join(rep.overlap[:8])
            return HealthCheckResult(
                passed=False,
                check_name="train_val_split_leakage",
                message=(
                    f"DATA LEAK ({rep.mode}, {rep.key_kind}-level): {rep.detail} "
                    f"train={rep.n_train} val={rep.n_val}. Overlap: {sample}"
                ),
                severity="error",
                category="data_leakage",
                yaml_keys=[
                    "data.index_path",
                    "data.validation_index_path",
                    "data.split.type",
                ],
                fix_hint=(
                    "Rebuild the split so no subject/file is in both train and "
                    "val. For mrixfields, regenerate with "
                    "`python scripts/data/build_mrixfields2026_manifest.py` (it "
                    "namespaces subjects by split); for a random split, ensure the "
                    "index groups all of a subject's records contiguously."
                ),
            )
        if rep.status == "skipped":
            return HealthCheckResult(
                passed=True,
                check_name="train_val_split_leakage",
                message=f"split-leakage check skipped ({rep.mode}): {rep.detail}",
                severity="info",
            )
        return HealthCheckResult(
            passed=True,
            check_name="train_val_split_leakage",
            message=(
                f"no train/val leakage ({rep.mode}): {rep.detail} "
                f"train={rep.n_train} val={rep.n_val}."
            ),
            severity="info",
        )

    def run_all_checks(self, config: TrainingSettings) -> HealthCheckReport:
        """Run all health checks and return aggregated report."""
        report = HealthCheckReport()
        report.results.extend(self.check_metric_backend_available(config))
        # Cluster-failure pre-flight guards (2026-07-07): contrast-id threading,
        # mrixfields pairing viability, and general train/val data-leak.
        report.results.append(self.check_contrast_conditioning_strategy_threaded(config))
        report.results.extend(self.check_mrixfields_pairing_viable(config))
        report.results.append(self.check_train_val_split_leakage(config))
        report.results.append(self.check_persistent_workers_on_short_epoch(config))

        # Required sections
        report.results.extend(self.check_required_sections(config))

        # Registry checks
        report.results.append(self.check_model_registry(config))
        # Phase 0 audit-ladder hardening (2026-05-22): close the
        # advertised-but-unresolvable failure mode that produced the
        # d8ccb8452 deletion ledger. See
        # TODO/deleted_model_types_reimplementation_plan.md Phase 0.
        report.results.append(self.check_registered_model_resolves(config))
        # #2255: resolving to a buildable class is not enough when the strategy
        # validates through the adapter's own reverse process — that method can
        # still be a stub, and the run finds out at the first eval interval.
        report.results.append(self.check_validation_path_is_wired(config))
        report.results.append(self.check_namespace_axis(config))
        report.results.extend(self.check_phase3_model_constraints(config))
        report.results.append(self.check_strategy_registry(config))

        # Domain alignment (pre-flight channel mismatch detection)
        report.results.extend(self.check_domain_alignment(config))

        # Loss and physics checks
        report.results.extend(self.check_loss_weight_declaration_ssot(config))
        report.results.extend(self.check_physics_config(config))

        # Tier-1 audit-ladder additions (errors)
        report.results.extend(self.check_advertised_options(config))
        report.results.extend(self.check_loss_domain_consistency(config))
        report.results.extend(self.check_amp_grad_clip_interaction(config))

        # Tier-1 warnings-and-fallbacks additions
        report.results.extend(self.check_early_stopping_metric_compatibility(config))
        # 2026-06 scientific-validation campaign: declared headline metric must be computed.
        report.results.extend(self.check_scientific_metadata(config))
        # 2026-07 ldm_two_stage triage: a vae_pretrain arm on paired data must
        # autoencode a single field, not train a translation direction.
        report.results.append(self.check_vae_pretrain_autoencodes_single_field(config))
        report.results.extend(self.check_metric_channel_compatibility(config))
        report.results.extend(self.check_declared_losses_registered(config))
        report.results.extend(self.check_paradigm_required_fields(config))

        # Tier-1 additions 2026-05-04 (smoke-test postmortem spec)
        report.results.append(self.check_denoising_model_channels(config))
        report.results.append(self.check_coil_processing_consistency(config))
        report.results.append(self.check_complex_unet_even_channels(config))
        report.results.append(self.check_model_contract_declared(config))
        report.results.append(self.check_dangerous_data_requires_contract(config))
        report.results.append(self.check_metric_domain_matches_loss_output(config))
        report.results.append(self.check_patch_size_power_of_two(config))
        report.results.append(self.check_hilbert_square_pow_two(config))
        report.results.append(self.check_pde_synthetic_datasets(config))
        report.results.append(self.check_acceleration_present(config))
        report.results.append(self.check_reveal_attribution_preconditions(config))
        report.results.append(self.check_acs_within_center_band(config))
        report.results.extend(self.check_nr_metrics_research_mode(config))
        report.results.append(self.check_val_batch_size(config))
        report.results.append(self.check_latent_decode_resolution(config))
        report.results.append(self.check_multi_contrast_model_support(config))
        report.results.append(self.check_vendor_model_support(config))
        # SPECTRA hardware-backend Tier-1 guards (implementation plan
        # Workstream C, C4). All no-op when backend_acceleration is absent.
        report.results.append(self.check_spectra_backend_schema(config))
        report.results.append(self.check_spectra_precision_envelope(config))
        report.results.append(self.check_spectra_gridding_sized(config))
        report.results.append(self.check_tta_adapter_only(config))
        report.results.append(self.check_metric_names_are_registered(config))
        report.results.append(self.check_repetition_count_is_achievable(config))
        report.results.append(self.check_transform_names_are_registered(config))
        report.results.append(self.check_declared_keys_are_not_discarded(config))
        report.results.append(self.check_component_kwargs_reach_constructor(config))
        report.results.append(self.check_declared_model_kwargs_are_read(config))
        report.results.append(self.check_kspace_scale_domain_is_declared(config))
        report.results.append(self.check_metric_transforms_agree(config))
        # `check_config_version_is_canonical` was deleted here along with the
        # fold it read. It searched the ledger for a `config_version`
        # VALUE_CHANGED_ON_FINALIZE record, and `_bind_config_version` was the
        # only emitter; with legacy versions refused outright the record can
        # never appear, so the check could only ever return "canonical" -- a
        # pass that measures nothing (pitfall #16). The loader now enforces the
        # same rule one layer earlier and strictly harder: a legacy version
        # cannot produce a TrainingSettings at all, so no audit can run on one.
        report.results.append(self.check_m4raw_nex_target_mode_declared(config))
        report.results.append(self.check_slice_level_records_queue_shape(config))
        report.results.extend(self.check_loss_domain_block_match(config))
        # Phase 2 of experiment-spec-card design: hard cross-check of
        # data-block ⇆ model registry capabilities, with declared adapters
        # honored as the only legitimate bridge. See
        # docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md
        report.results.append(self.check_data_model_compatibility(config))
        # Phase 5: catch model.output_domain ≠ losses.output_domain when
        # there's no auto-bridge. The two LossBuilder auto-bridge
        # directions (image→kspace, image→complex) are tolerated until
        # the Phase 4d migration promotes them to explicit adapters.
        report.results.append(self.check_model_loss_output_domain(config))
        # E-VIZ2 (smoke 2026-06-16): target_domain (Priority-1 in domain
        # inference) overriding the registered output_domain → spurious IFFT on
        # viz → k-space-rendered-as-image. The check above reads the registered
        # domain, not target_domain, so this gap needed its own check.
        report.results.append(self.check_target_domain_matches_registered_output_domain(config))
        # Attention/backbone/feature-domain compatibility for the
        # kspace_cold_diffusion family (pitfall #16 facade + build-time crashes
        # from a domain-incompatible or seam-less attention request).
        report.results.extend(self.check_attention_domain_compatibility(config))
        # Phase 8: transforms + physics contracts.
        report.results.append(self.check_sfc_scan_mode_matches_spatial_dims(config))
        report.results.append(self.check_data_consistency_requires_kspace(config))
        report.results.append(self.check_acceleration_consistency(config))
        report.results.append(self.check_acceleration_schedule_steps_match_diffusion(config))
        report.results.append(self.check_validation_cascade_levels_in_range(config))
        report.results.append(self.check_timesteps_vs_step_buckets(config))
        report.results.append(self.check_validation_split_redundancy(config))
        report.results.append(self.check_dc_method_physics_consistency(config))
        report.results.append(self.check_dc_knobs_inert_by_method(config))
        report.results.append(self.check_time_keys_within_num_timesteps(config))
        # F15 (2026-05-21 smoke audit): discriminator-declared YAMLs need losses.gan.
        report.results.append(self.check_discriminator_requires_gan_losses(config))

        # Phase 10 (2026-05-05): normalization correctness audits.
        # See docs/audits/2026-05-05-silent-correctness-audit.md §F.
        report.results.append(self.check_normalization_kspace_compatibility(config))
        report.results.append(self.check_normalize_kspace_image_norm_mutex(config))
        report.results.append(self.check_coil_flatten_image_norm_trap(config))
        report.results.append(self.check_kspace_percentile_range(config))
        report.results.append(self.check_pin_memory_no_cuda(config))
        # Phase 11 (2026-05-05 round 2): schema cross-field validators.
        report.results.append(self.check_acceleration_bounds(config))
        report.results.append(self.check_strategy_class_resolves(config))
        report.results.append(self.check_adversarial_strategy_requires_gan_loss(config))
        report.results.append(self.check_cycle_bloch_requires_discriminator(config))
        # PR-0 (2026-05-28): themed generic-base aliases must wire a registered
        # themed component, else they silently run as a vanilla baseline (#9).
        report.results.append(self.check_themed_strategy_requires_themed_component(config))
        # Mamba/SSM models require the official mamba_ssm kernel (the GRU
        # fallback is NOT an SSM) — catch a missing kernel at audit time.
        report.results.append(self.check_mamba_models_require_mamba_ssm(config))
        report.results.append(self.check_lion_learning_rate_scale(config))
        report.results.append(self.check_compile_with_sharded_strategy(config))
        report.results.append(self.check_deepspeed_extra_installed(config))
        report.results.append(self.check_deepspeed_precision_coherent(config))
        # Diffusion arms train in fp32. 12 arms resolved AMP ON when this landed
        # -- invisible to a grep because the legacy `optimization.use_amp` folds
        # onto `optimization.precision.enabled` (PR #809).
        report.results.append(self.check_diffusion_precision_policy(config))
        # Inductor cannot codegen complex operators. It does not fail -- it
        # falls back to eager and warns once, so a compiled complex arm reports
        # throughput that belongs to a different configuration (pitfall #16).
        report.results.append(self.check_bf16_requires_ampere(config))
        report.results.append(self.check_compile_with_complex_model(config))
        report.results.append(self.check_deepspeed_topology_coherent(config))
        # The world-size axis of the same question: a stage that owns the state
        # still shards it across nobody at one rank, and pays the bucket/hook
        # cost to do so. Advisory (info) by construction -- num_devices is a
        # defaulted, launcher-overwritten value, so warning on it would fire on
        # every launcher-driven arm with no config-side fix.
        report.results.append(self.check_deepspeed_zero_stage_has_ranks_to_shard(config))
        report.results.append(self.check_zenflow_accumulation_conflict(config))
        report.results.append(self.check_deepcompile_supported(config))
        report.results.append(self.check_optimizer_registered(config))
        report.results.append(self.check_deepspeed_consolidated_best_checkpoint(config))
        # F7 (round 4 2026-05-17): surface check_domain_alignment blind
        # spots as warnings so --strict mode can hard-gate configs that
        # would otherwise crash at the first training batch with
        # [DomainMismatch]. See TODO/audit/smoke_audit_20260516.md §F7.
        report.results.extend(self.check_channel_audit_assumptions(config))
        # F7-Hoist (round 6 2026-05-17): static adapter-chain channel
        # resolution — promotes the round-4 F7 warning to a hard error
        # when all chain steps have known channel effects.
        report.results.append(self.check_adapter_chain_channel_resolution(config))
        # F2 (round 6 2026-05-17): visualization-interval reachability —
        # flags the 147 silent-regression pattern where the val loop
        # ends before any viz tick fires.
        report.results.append(self.check_visualization_interval_reachable(config))
        # F5 (round 6 2026-05-17): reject hardcoded /project/ and
        # /scratch/ paths in manifest fields; force PathResolver use.
        report.results.append(self.check_hardcoded_cluster_paths(config))
        # F-OUT (round 6 2026-05-17): enforce
        # ``experiments/results/<name>`` convention for training.output_dir
        # so smoke tooling can auto-discover mosaics + reports. See
        # TODO/audit/smoke_audit_20260516.md §F-OUT.
        report.results.append(self.check_output_dir_convention(config))
        # #1929: the convention check above is a PREFIX test and cannot see an
        # arm whose output_dir, checkpoints and logs name different experiments.
        # Checkpoint selection is declared two ways and only one is live; 75 arms
        # in the corpus disagree with early stopping enabled (non-negotiable 17).
        report.results.append(self.check_best_metric_matches_early_stopping(config))
        report.results.append(self.check_identity_paths_agree(config))
        report.results.append(self.check_epochs_max_iterations_mutex(config))

        # v6.1 — paradigm-expansion checks
        report.results.append(self.check_concomitant_requires_low_field(config))
        report.results.append(self.check_field_strength_declared(config))
        # PMPS Phase-0 (Bloch-grounded acquisition encoder).
        report.results.append(self.check_bloch_grounded_requires_reference_panel(config))
        report.results.append(self.check_acquisition_params_required_for_bloch_grounded(config))
        report.results.append(self.check_concomitant_required_at_ulf(config))
        # MICCAI MRIxFields2026 idea 4.2 — cocycle unified-operator Tier-1 guards
        # (degenerate-solution + reference-range + single-model anti-ensemble).
        report.results.extend(self.check_field_cocycle_arm(config))
        # MICCAI MRIxFields2026 idea 2.1 — relaxometry+Bloch synthesis Tier-1 guards
        # (J>=3 identifiability + in_channels match + dispersion envelope).
        report.results.extend(self.check_bloch_synth_arm(config))
        # Curriculum anti-facade: every loss_schedule target must be consumed.
        report.results.append(self.check_curriculum_targets_consumed(config))
        # Curriculum runtime-safety: every target's base weight must RESOLVE.
        report.results.append(self.check_curriculum_targets_resolvable(config))
        # PMPS Phase-1/2/3/4 paradigm-specific checks.
        report.results.append(
            self.check_tissue_diffusion_pretrain_requires_field_strength_conditioning(config)
        )
        report.results.append(self.check_corruption_calibration_requires_tissue_prior(config))
        report.results.append(self.check_paired_synthesis_requires_prior_and_calibration(config))
        report.results.append(self.check_data_efficiency_harness_settings_well_formed(config))
        # Phase-1 (VF-Residual conformal) Tier-1 checks.
        report.results.append(self.check_vf_residual_marker_matrix_well_conditioned(config))
        report.results.append(self.check_vf_residual_exchangeability_precondition(config))
        # Phase-2 (IB-VF / InfoNCE) Tier-1 checks.
        report.results.append(self.check_infonce_effective_batch_size_sufficient(config))
        report.results.append(self.check_ib_beta_positive(config))
        report.results.append(self.check_nav_encoder_input_is_dc_kspace(config))
        # Conditioning guard (2026-05-26): model.conditioning sources must be
        # registered encoders AND consumed by the strategy (no silent no-op).
        report.results.append(self.check_conditioning_sources_supported(config))
        # Phase-3 (Twin-Likelihood DPS) Tier-1 checks.
        report.results.append(self.check_twin_dps_guidance_scales_finite_and_positive(config))
        report.results.append(self.check_phase_stego_basis_compatible(config))
        report.results.append(self.check_chd_calibration_corpus_size(config))
        report.results.append(self.check_pathology_test_set_size(config))
        report.results.append(self.check_dp_compatible_optimizer(config))
        report.results.append(self.check_validation_badge_gates_complete(config))
        report.results.append(self.check_acceleration_adaptive_target_rate_consistent(config))
        report.results.append(self.check_pilot_hardware_constraints_sane(config))
        # VF campaign Phase 1 (plan I-5 / CW-2): marker conditioning + artefact
        # existence. Both info-skip when not applicable, so they add no
        # false-positive load to the 33 synthetic-injection arms.
        report.results.append(self.check_marker_subspace_conditioning(config))
        report.results.append(self.check_checkpoint_existence(config))
        report.results.append(self.check_b0_field_metric_requires_hz_model(config))
        report.results.append(self.check_dds_requires_score_model(config))
        report.results.append(self.check_ei_sensing_margin(config))
        report.results.append(self.check_ei_robust_key_matches_correction(config))
        report.results.append(self.check_ssdu_selection_density_range(config))
        report.results.append(self.check_robust_ssdu_key_matches_correction(config))
        report.results.append(self.check_ambient_requires_score_model(config))
        report.results.append(self.check_operator_posterior_requires_complex(config))
        report.results.append(
            self.check_trajectory_metric_requires_matching_parametrization(config)
        )
        report.results.append(self.check_trajectory_recon_requires_measured_emit(config))

        # Operator-ID (Proposal 1) paradigm checks (only fire when
        # training.operator_id is present).
        report.results.extend(self.check_operator_id_config(config))
        # Contrast/field-agnostic bundle (2026-06-29 design): M2 MCGI, M3 LCAH,
        # M4 DL-BAE. Each no-ops (empty list / info-skip) when its block is
        # absent, so they add no load to unrelated arms.
        report.results.append(self.check_mcgi_invariance_declared(config))
        report.results.extend(self.check_acq_hypernetwork_config(config))
        report.results.extend(self.check_dispersion_bloch_ae_config(config))
        # Workflow contract (imaging-regime × task): error when the
        # ``workflow:`` block is absent, names a STUB regime the framework
        # cannot run, or pairs a task the regime does not support.
        report.results.append(self.check_workflow_declared(config))
        report.results.append(self.check_workflow_required_axes(config))
        report.results.append(self.check_workflow_spatial_rank(config))
        report.results.append(self.check_workflow_signal_domain(config))
        report.results.append(self.check_workflow_dataset_signal_domain(config))
        report.results.append(self.check_workflow_component_regime(config))
        report.results.extend(self.check_knob_applicability(config))
        # The fiducial must calibrate INSIDE the tissue range it certifies.
        # Written and unit-tested on 2026-07-26 but never wired here, so it had
        # never rejected anything: its three tests call the method directly, which
        # proves the check works, not that the pipeline runs it. Found by
        # `meta.health_checker_no_orphan_checks`.
        report.results.append(self.check_marker_kappa_in_tissue_range(config))

        # Note (2026-05-05): the Phase-9 audits
        # `check_svd_compression_phase_safety` and
        # `check_validation_image_domain_safe` are NOT wired into the bulk
        # audit because:
        #
        #   * The SVD silent-skip is now a hard ValueError in
        #     SVDCoilCompressionTransform.apply_transform (see
        #     src/data/transforms/coil_compression.py), so any
        #     real-stacked SVD invocation crashes immediately at the
        #     transform — no need for a config-load-time guard.
        #   * The validation-image doubled-brain bug was fixed in
        #     src/infrastructure/training/strategies/diffusion.py
        #     (the unconditional .abs() before kspace_to_image was
        #     replaced with a is_image_domain-only guard), so the
        #     visualization is now correct regardless of
        #     compute_image_metrics.
        #
        # The check methods themselves are retained for future use
        # (e.g. as opt-in pre-flight checks at experiment design time)
        # but firing them on every config produced a 113/61 false-positive
        # storm without flagging any genuine misconfigurations.

        return report


def validate_config_health(
    config: TrainingSettings, *, log_summary: bool = True
) -> HealthCheckReport:
    """Convenience function to validate config at pipeline start.

    Runs all health checks and logs a summary. Callers can inspect
    ``report.passed`` for an overall pass/fail and ``report.errors`` /
    ``report.warnings`` for actionable details (e.g. to filter for
    domain-alignment errors and abort the pipeline before bootstrap).

    Args:
        log_summary: emit the ``Config Health: n/m checks passed`` line and the
            per-failure records. Pass ``False`` when the caller republishes the
            results through its own surface, so the same checks are not narrated
            twice in one process. A train run reaches this function twice
            — once from ``pipelines/train.py``'s fail-fast gate and once from the
            witness registry in ``bootstrap`` — which is why the summary appeared
            duplicated in every job log.
    """
    checker = ConfigHealthChecker()
    report = checker.run_all_checks(config)
    if log_summary:
        report.log_summary()
    return report
