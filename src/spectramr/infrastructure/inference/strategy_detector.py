"""Strategy type detection from configuration.

Declarative rule-based pattern matching to determine inference strategy type
from configuration dictionaries. Reduces cyclomatic complexity from if/elif
chains to O(1) lookups with priority-ordered rules.
"""

from collections.abc import Callable
from typing import Any


class StrategyDetectionRule:
    """Single rule for detecting a strategy type from config.

    A rule specifies:
    - How to extract a value from config (via config_path or extractor function)
    - How to test if the value matches (via pattern matching or custom checker)
    - What strategy name to return if matched
    """

    def __init__(
        self,
        strategy_name: str,
        config_path: str | None = None,
        value_pattern: str | None = None,
        value_equals: Any | None = None,
        contains: str | None = None,
        extractor: Callable[[dict], Any] | None = None,
        checker: Callable[[Any], bool] | None = None,
    ):
        """
        Args:
            strategy_name: Strategy type to return if rule matches.
            config_path: Dot-separated path to config value, e.g., "training.training_mode".
            value_pattern: Exact string to match (case-insensitive).
            value_equals: Exact value to match (any type).
            contains: Substring to check (case-insensitive).
            extractor: Custom function to extract value from config dict.
            checker: Custom function to check if extracted value matches.

        Note:
            Exactly one of (config_path, extractor) must be provided.
            Exactly one of (value_pattern, value_equals, contains, checker) must be provided.
        """
        self.strategy_name = strategy_name
        self.config_path = config_path
        self.value_pattern = value_pattern
        self.value_equals = value_equals
        self.contains = contains
        self.extractor = extractor
        self.checker = checker

    def matches(self, config: dict[str, Any]) -> bool:
        """Test if this rule matches the given config.

        Args:
            config: Configuration dictionary to test.

        Returns:
            True if rule matches, False otherwise.
        """
        # Extract value from config
        if self.extractor is not None:
            try:
                value = self.extractor(config)
            except (KeyError, TypeError, AttributeError):
                return False
        elif self.config_path is not None:
            value = self._extract_by_path(config, self.config_path)
            if value is None:
                return False
        else:
            return False

        # Check if value matches pattern
        if self.checker is not None:
            return self.checker(value)
        elif self.value_pattern is not None:
            return str(value).lower() == self.value_pattern.lower()
        elif self.value_equals is not None:
            return value == self.value_equals
        elif self.contains is not None:
            return self.contains.lower() in str(value).lower()
        else:
            return False

    @staticmethod
    def _extract_by_path(config: dict, path: str) -> Any:
        """Extract value from nested config using dot-separated path.

        Args:
            config: Configuration dictionary.
            path: Dot-separated path, e.g., "training.training_mode".

        Returns:
            Extracted value or None if path doesn't exist.
        """
        keys = path.split(".")
        current = config
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    return None
            else:
                return None
        return current


class StrategyDetector:
    """Detects inference strategy type from configuration using priority-ordered rules.

    Replaces complex if/elif chains (CC=31) with declarative rule lookup (CC=2).
    Rules are evaluated in order of registration; first match wins.
    """

    def __init__(self):
        """Initialize detector with empty rule list."""
        self.rules: list[StrategyDetectionRule] = []

    def add_rule(self, rule: StrategyDetectionRule) -> None:
        """Add a detection rule.

        Args:
            rule: Rule to add. Rules are evaluated in order added.
        """
        self.rules.append(rule)

    def detect(self, config: dict[str, Any], default: str = "reconstruction") -> str:
        """Detect strategy type from config.

        Args:
            config: Configuration dictionary.
            default: Default strategy if no rules match.

        Returns:
            Detected strategy type name.
        """
        for rule in self.rules:
            if rule.matches(config):
                return rule.strategy_name
        return default


def create_default_detector() -> StrategyDetector:
    """Create detector with default strategy detection rules.

    Rules are ordered by priority, and the ordering principle is **declared
    specialisation before declared generalisation, and every declaration before
    every heuristic**:

    0. Explicit sub-paradigm declaration (``training.diffusion.type: cold``)
    1. Explicit ``training.training_mode``
    2-3. Model-type string patterns -- HEURISTICS, see the warning below
    4. Presence of a config section (``gan``, ``vae``, ...)
    5. Specific model config parameters (``latent_dim``, ``style_dim``)

    Priority 0 exists because ``cold_diffusion`` REFINES ``diffusion``. Without
    it the generic ``training_mode: diffusion`` rule matched first and shadowed
    every cold arm: 87 arms had a cold rule that matched and lost to a generic
    one, including ``experiment_11_kspace_cold_diffusion.yaml`` itself. The
    docstring claimed "most specific first" while the order was by config
    SOURCE, which is not the same thing (issue #991).

    **``model_type`` is an architecture, not a paradigm**, which is why the
    priority-2/3 rules must stay BELOW ``training_mode`` and were not hoisted
    with the fix. ``exp_205_kspace_gan_spectral.yaml`` declares
    ``training_mode: gan`` with ``GANTrainingStrategy`` and merely uses
    ``kspace_cold_diffusion_generator`` as its generator network: it is a GAN,
    and a rule that read the architecture as the paradigm would misroute it.

    Returns:
        Configured StrategyDetector instance.
    """
    detector = StrategyDetector()

    # Priority 0: an explicit cold-diffusion DECLARATION beats the generic
    # `training_mode: diffusion` that would otherwise swallow it.
    #
    # This replaces a rule that read `cfg.get("diffusion")` against the WHOLE
    # `config.model_dump()` -- i.e. the top-level `config.diffusion`, a
    # different field from `config.training.diffusion`, deprecated, and `None`
    # on every modern arm ("kept solely to avoid breaking legacy YAMLs under
    # extra='forbid'"). It therefore inspected a block the canonical spelling
    # does not populate. Deleting it changes no arm's routing once this rule
    # exists -- verified across all 725 constructible arms.
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="cold_diffusion",
            extractor=lambda cfg: (cfg.get("training") or {}).get("diffusion") or {},
            checker=lambda diff_cfg: (
                isinstance(diff_cfg, dict)
                and (
                    str(diff_cfg.get("type") or "").lower() in {"cold", "cold_diffusion"}
                    or "cold" in str(diff_cfg.get("sampler") or "").lower()
                )
            ),
        )
    )

    # Priority 1: Explicit training.training_mode (most reliable)
    for mode_name in [
        "disentangled",
        "vqvae",
        "physics_driven",
        "domain_adaptation",
        "latent_gan",
        "latent_diffusion",
        "cascaded_diffusion",
        "mae",
        "vae",
        "diffusion",
    ]:
        detector.add_rule(
            StrategyDetectionRule(
                strategy_name=mode_name,
                config_path="training.training_mode",
                value_pattern=mode_name,
            )
        )

    # Special case: 'adversarial' maps to 'gan'
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="gan",
            config_path="training.training_mode",
            value_pattern="adversarial",
        )
    )
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="gan",
            config_path="training.training_mode",
            value_pattern="gan",
        )
    )

    # Priority 2: model_type string patterns (root level)
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="disentangled",
            config_path="model_type",
            contains="disentangled",
        )
    )
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="cold_diffusion",
            config_path="model_type",
            contains="cold_diffusion",
        )
    )
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="cold_diffusion",
            config_path="model_type",
            contains="cold_mri",
        )
    )

    # Priority 3: model.model_type string patterns
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="disentangled",
            config_path="model.model_type",
            contains="disentangled",
        )
    )
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="cold_diffusion",
            config_path="model.model_type",
            contains="cold_diffusion",
        )
    )
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="cold_diffusion",
            config_path="model.model_type",
            contains="cold_mri",
        )
    )

    # Priority 4: Explicit config sections
    # Check for disentangled section
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="disentangled",
            extractor=lambda cfg: "disentangled" in cfg,
            checker=lambda exists: exists,
        )
    )

    # The cold sub-check that used to live here read the deprecated top-level
    # `config.diffusion` and has moved to priority 0, reading
    # `training.diffusion`. See that rule's comment for why.
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="diffusion",
            extractor=lambda cfg: "diffusion" in cfg,
            checker=lambda exists: exists,
        )
    )

    # Check for gan section
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="gan",
            extractor=lambda cfg: "gan" in cfg,
            checker=lambda exists: exists,
        )
    )

    # Check for vae section
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="vae",
            extractor=lambda cfg: "vae" in cfg,
            checker=lambda exists: exists,
        )
    )

    # Priority 5: model_type string patterns (fallback, broader matching)
    detector.add_rule(
        StrategyDetectionRule(strategy_name="gan", config_path="model_type", contains="gan")
    )
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="diffusion", config_path="model_type", contains="diffusion"
        )
    )
    detector.add_rule(
        StrategyDetectionRule(strategy_name="vae", config_path="model_type", contains="vae")
    )

    # Priority 6: Specific model config parameters
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="vae",
            extractor=lambda cfg: "latent_dim" in cfg.get("model", {}),
            checker=lambda exists: exists,
        )
    )
    detector.add_rule(
        StrategyDetectionRule(
            strategy_name="disentangled",
            extractor=lambda cfg: "style_dim" in cfg.get("model", {}),
            checker=lambda exists: exists,
        )
    )

    return detector
