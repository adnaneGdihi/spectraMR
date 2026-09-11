#!/usr/bin/env python3
"""Integrity tests for model registry.

Scans source code to verify:
- No duplicate @register_model decorators
- All registrations have valid names and training_modes
- Registry matches source code
"""

import ast
import re
from collections import defaultdict
from pathlib import Path

import pytest


class RegistrationScanner:
    """Scans source code for model registrations."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.registrations: dict[str, list[tuple[str, int]]] = defaultdict(list)

    @staticmethod
    def _registered_name(decorator: ast.expr) -> str | None:
        """The model name a single ``@register_model(...)`` decorator declares.

        Accepts BOTH forms the tree actually uses -- ``name="x"`` in any keyword
        position, and the positional ``@register_model("x", "y")`` idiomatic in
        older generators like ``pma_varnet``. Returns ``None`` for anything that
        is not a ``register_model`` call, or whose name is not a literal.
        """
        if not isinstance(decorator, ast.Call):
            return None
        func = decorator.func
        # Matches a bare ``register_model(...)`` and a qualified
        # ``registry.register_model(...)`` alike.
        called = getattr(func, "id", None) or getattr(func, "attr", None)
        if called != "register_model":
            return None
        for keyword in decorator.keywords:
            if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                value = keyword.value.value
                return value if isinstance(value, str) else None
        if decorator.args and isinstance(decorator.args[0], ast.Constant):
            value = decorator.args[0].value
            return value if isinstance(value, str) else None
        return None

    def scan_file(self, file_path: Path):
        """Scan a Python file for @register_model decorators.

        Parsed with ``ast``, not matched with a regex. The regex this replaced
        required ``name=`` to be the FIRST keyword
        (``@register_model\\s*\\(\\s*name\\s*=``), which stopped being true the moment a
        change put ``role=`` first and ruff-format broke the call across lines.
        Measured at the time: the regex saw 492 registrations where the tree has
        513, missing 21 -- every discriminator, because that is where ``role=``
        leads. It saw nothing the AST does not, so the blindness was pure loss.

        The failure mode is what makes this worth parsing properly: a scanner
        that under-counts does not go red. ``get_duplicates`` simply stops
        reporting duplicates among the names it cannot see, and the sibling
        census test asserts *static names are live*, which a smaller static set
        satisfies trivially. Only the capability-parity test happened to fail
        loudly. A detector that narrows in silence is the shape NN15 is about.

        ``ast.walk`` rather than a scan of ``tree.body``: registrations are not
        all module-level -- some sit inside a function that builds a family of
        variants -- and a top-level-only walk would reintroduce a blind spot of
        its own.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            return

        relative_path = str(file_path.relative_to(self.base_path))
        for node in ast.walk(tree):
            for decorator in getattr(node, "decorator_list", []):
                model_name = self._registered_name(decorator)
                if model_name is not None:
                    self.registrations[model_name].append((relative_path, decorator.lineno))

    def scan_directory(self, directory: Path):
        """Recursively scan ``directory`` for registrations, skipping test files.

        The skip is matched against the path RELATIVE to the scan root. Matching
        the absolute path -- ``if "test" not in str(py_file)``, which this
        replaced -- makes the scanner a property of where the repository happens
        to be checked out: under ``~/testing/`` or a ``pytest`` ``tmp_path``,
        every file is skipped, ``registrations`` is empty, and every assertion in
        this file passes over nothing. That is not a hypothetical -- it is how
        the plant below first came back green-on-empty.
        """
        for py_file in directory.rglob("*.py"):
            relative = py_file.relative_to(directory)
            if not any("test" in part for part in relative.parts):
                self.scan_file(py_file)

    def get_duplicates(self) -> dict[str, list[tuple[str, int]]]:
        """Return models registered multiple times."""
        return {
            name: locations
            for name, locations in self.registrations.items()
            if len(locations) > 1
        }

    def get_all_registrations(self) -> dict[str, list[tuple[str, int]]]:
        """Return all registrations."""
        return dict(self.registrations)


#: One planted registration per SHAPE the scanner must see, with the line it sits on.
#:
#: NN15: a gate is only a gate for the violation shape it has been watched to fail on.
#: The regex this scanner replaced had gone green for years on shape 2 while being blind
#: to the other three, so "it passes" was never evidence. Shape 1 is the one that
#: actually broke it -- ``role=`` ahead of ``name=``, split across lines by ruff-format.
_PLANTED_SHAPES = """
from spectramr.models.registry import register_model


@register_model(
    role="discriminator", name="PLANT_role_first", training_mode="gan",
)
class _RoleFirst:
    pass


@register_model(name="PLANT_name_first", role="discriminator")
class _NameFirst:
    pass


@register_model("PLANT_positional", "gan")
class _Positional:
    pass


def _build_family():
    @register_model(role="discriminator", name="PLANT_function_local")
    class _FunctionLocal:
        pass
"""


class TestTheScannerSeesEveryDecoratorShape:
    """The scanner is exercised against planted registrations, not only the tree.

    Planted rather than asserted against a count: a count pinned to today's tree
    goes stale on the next registration and gets bumped without being read, which
    is how the regex's 21-name blind spot survived. A shape either parses or it
    does not, and that answer does not drift.
    """

    @pytest.fixture
    def planted(self, tmp_path: Path) -> "RegistrationScanner":
        (tmp_path / "planted_models.py").write_text(_PLANTED_SHAPES, encoding="utf-8")
        scanner = RegistrationScanner(tmp_path)
        scanner.scan_directory(tmp_path)
        return scanner

    @pytest.mark.parametrize(
        "name,lineno",
        [
            ("PLANT_role_first", 5),
            ("PLANT_name_first", 12),
            ("PLANT_positional", 17),
            ("PLANT_function_local", 23),
        ],
    )
    def test_the_shape_is_seen_at_its_own_line(self, planted, name: str, lineno: int) -> None:
        """Each shape is found, and attributed to the DECORATOR's line.

        The line number matters because it is what a duplicate report points a
        reader at; a decorator span reported at the class's line sends them to
        the wrong one of four stacked registrations.
        """
        assert name in planted.registrations, (
            f"{name} is declared in the planted file and the scanner did not see it — "
            f"it saw {sorted(planted.registrations)}"
        )
        assert planted.registrations[name] == [("planted_models.py", lineno)]

    def test_the_plant_would_catch_the_regression_it_was_written_for(self, planted) -> None:
        """The retired regex fails this plant, so the plant discriminates.

        Without this leg the plant could be satisfied by any scanner at all,
        including the broken one, and would document nothing.
        """
        retired = re.compile(r'@register_model\s*\(\s*name\s*=\s*["\']([^"\']+)["\']')
        assert sorted(retired.findall(_PLANTED_SHAPES)) == ["PLANT_name_first"], (
            "the retired regex no longer misses the three shapes this plant exists to "
            "cover — re-derive the plant rather than deleting this assertion"
        )
        assert len(planted.registrations) == 4


class TestRegistryIntegrity:
    """Test registry integrity by scanning source code."""

    @pytest.fixture
    def scanner(self):
        """Create scanner for src/models directory."""
        # __file__ = tests/unit/models/test_registry_integrity.py
        # parents[3] = repo root
        base_path = Path(__file__).resolve().parents[3]
        models_path = base_path / "src" / "spectramr" / "models"

        scanner = RegistrationScanner(base_path)
        scanner.scan_directory(models_path)
        return scanner

    # Names that are intentionally registered BOTH in the per-file
    # baseline implementation AND the v6.1 metadata manifest
    # (src/models/_v6_1_registrations.py). The manifest enrichment is a
    # known design point — the last-write-wins behaviour of the
    # registry promotes the enriched (input_domain, output_domain,
    # accepts_complex) record over the bare per-file one.
    KNOWN_DUAL_REGISTERED = {
        "cdiffmr_baseline",
        "shen2024_baseline",
        "fdb_baseline",
    }

    def test_no_duplicate_registrations(self, scanner):
        """Verify no model name is registered twice (except known dual sites)."""
        duplicates = scanner.get_duplicates()
        unexpected = {
            name: locs
            for name, locs in duplicates.items()
            if name not in self.KNOWN_DUAL_REGISTERED
        }

        if unexpected:
            error_msg = "Found duplicate model registrations:\n"
            for name, locations in unexpected.items():
                error_msg += f"\n'{name}' registered {len(locations)} times:\n"
                for file_path, line_num in locations:
                    error_msg += f"  - {file_path}:{line_num}\n"

            pytest.fail(error_msg)

    def test_minimum_registrations(self, scanner):
        """Verify we have minimum expected registrations."""
        all_registrations = scanner.get_all_registrations()
        total = len(all_registrations)

        assert total >= 100, f"Expected at least 100 registrations, found {total}"

    def test_newly_registered_models_present(self, scanner):
        """Verify our newly registered models are in source code.

        Names migrated to the ``vae_*`` / ``latent_*`` prefixes during
        the schema-6.0 refactor; pinning the *current* names so a
        future rename surfaces here.
        """
        expected_models = [
            "vae_fourier_kan_vae",
            "vae_wavelet_kan_vae",
            "sparse_vae",
            "vqvae_generator",
            "vqgan",
            "configurable_vae",
            "patch_latent_discriminator",
        ]

        all_registrations = scanner.get_all_registrations()

        for model_name in expected_models:
            assert (
                model_name in all_registrations
            ), f"Expected model '{model_name}' not found in source code"

    def test_no_commented_duplicates(self, scanner):
        """Verify we don't have both registered and commented registrations."""
        # This would need more sophisticated parsing
        # For now, we verify unique registrations
        all_registrations = scanner.get_all_registrations()

        # Each model should appear exactly once
        for name, locations in all_registrations.items():
            if len(locations) > 1:
                # Check if any are commented
                base_path = Path(__file__).resolve().parents[3]

                for file_path, line_num in locations:
                    full_path = base_path / file_path
                    with open(full_path) as f:
                        lines = f.readlines()
                        line = lines[line_num - 1]

                        # If line starts with #, it's commented
                        if line.strip().startswith("#"):
                            pytest.fail(
                                f"Model '{name}' has both active and commented "
                                f"registrations at {file_path}:{line_num}"
                            )


class TestRegistrationFormat:
    """Test that registrations follow correct format."""

    def test_registration_includes_training_mode(self):
        """Verify all registrations specify training_mode.

        Accepts both keyword form ``@register_model(name="x", training_mode="y")``
        and positional form ``@register_model("x", "y", ...)`` — the latter is
        idiomatic in older generators like ``pma_varnet``.
        """
        base_path = Path(__file__).resolve().parents[3]
        models_path = base_path / "src" / "spectramr" / "models"

        # Capture the decorator call's argument span (DOTALL because the
        # decorator commonly spans multiple lines with kwargs). Anchored to
        # line-start (MULTILINE ``^\s*``) so a ``@register_model`` mentioned
        # inside a trailing ``# noqa`` comment is not mis-parsed as a call.
        full_pattern = re.compile(
            r"^\s*@register_model\s*\(\s*(?P<args>.*?)\s*\)",
            re.DOTALL | re.MULTILINE,
        )
        # Positional form: two string literals as the first two args.
        positional_form = re.compile(
            r"""^\s*["'][^"']+["']\s*,\s*["'][^"']+["']""",
            re.VERBOSE,
        )

        issues = []

        for py_file in models_path.rglob("*.py"):
            try:
                with open(py_file) as f:
                    content = f.read()
            except Exception:
                continue

            for match in full_pattern.finditer(content):
                args = match.group("args")
                has_kw = "training_mode" in args
                has_pos = bool(positional_form.match(args))
                if not (has_kw or has_pos):
                    line_num = content[: match.start()].count("\n") + 1
                    relative_path = str(py_file.relative_to(base_path))
                    issues.append(f"{relative_path}:{line_num}")

        if issues:
            pytest.fail(
                f"Found {len(issues)} registrations without training_mode:\n"
                + "\n".join(f"  - {issue}" for issue in issues[:10])
            )

    def test_valid_training_modes(self):
        """Verify all training_modes are valid.

        Scans both keyword and positional forms of ``@register_model``.
        """
        base_path = Path(__file__).resolve().parents[3]
        models_path = base_path / "src" / "spectramr" / "models"

        # Pinned universe of decorator-form registration modes (the paradigm
        # dispatch bucket, distinct from config VALID_TRAINING_MODES). This is
        # a source-scan ratchet: adding a new mode string is a deliberate act
        # that must be reflected here, so a typo'd mode fails the gate. Refreshed
        # 2026-07-02 after the src->spectramr rename left this test scanning a
        # non-existent path (silently vacuous) — the set had drifted to 24 while
        # the tree carried 52.
        # 2026-07-02 (sota_registry retirement): "acquisition" and "synthesis"
        # added — the retired module registered rl_mri / gen_3dgs / causal_gan
        # under these modes via CALL-form register_model(...) which this
        # decorator-scan never saw; the migration to @register_model decorators
        # makes them visible here. The census pins the exact (name, mode) pairs.
        valid_modes = {
            "acq_hypernetwork",
            "acquisition",
            "acquisition_learning",
            "active_acquisition",
            "adaptation",
            "bloch_field",
            # Pre-existing red on dev, repaired here: these three are registered
            # on live models and all dispatch, but this list (a fourth
            # hand-maintained copy of the mode names) had drifted behind them.
            "bloch_synth",
            "field_cocycle",
            "stargan_v2",
            "brenier_synthesis",
            "cold_diffusion",
            "compression",
            "corruption_calibration",
            "cross_field_translation",
            "diffusion",
            # 2026-08-07: DL-BAE (contrast/field-agnostic bundle M4) registers
            # `disp_bloch_ae` under this mode. `acq_hypernetwork` above was
            # already pinned by lcah_encoder, whose STRATEGY_CLASS_PATHS key
            # landed in the same change.
            "dispersion_bloch_ae",
            "domain_adaptation",
            "doob_bridge",
            "encoder",
            "experimental",
            "federated",
            "field_conditioned_inr",
            "field_flow",
            "field_fno",
            "field_guided_diffusion",
            "field_wiener",
            "fisher_rao_geodesic",
            "flow_matching",
            "gan",
            "generative",
            "geomamba_ulf",
            "ib_vf",
            "koopman_field",
            "learnable_acquisition",
            "lora_modulation",
            "mccann_field_path",
            "meta_learning",
            "monotone_field",
            "multi_param_mapping",
            "operator_id",
            "paired_synthesis",
            # siren_sens_net retagged here 2026-07-19 from the removed
            # "sensitivity_estimation" orphan (it named no strategy).
            "pinn",
            "reconstruction",
            "recoverability_vib",
            "scattering_besov",
            "sparse_frame",
            "ssl",
            "steerable_synthesis",
            "supervised",
            "synthesis",
            "tissue_diffusion_pretrain",
            "trajectory_recon",
            "transport",
            "utility",
            "vae",
            "virtual_fiducial",
            "vqvae",
            "wrapper",
        }

        kw_pattern = re.compile(r'training_mode\s*=\s*["\']([^"\']+)["\']')
        pos_pattern = re.compile(
            r"""^\s*@register_model\s*\(\s*["'][^"']+["']\s*,\s*["']([^"']+)["']""",
            re.DOTALL | re.MULTILINE,
        )

        invalid_modes = []

        for py_file in models_path.rglob("*.py"):
            try:
                with open(py_file) as f:
                    content = f.read()
            except Exception:
                continue

            for pattern in (kw_pattern, pos_pattern):
                for match in pattern.finditer(content):
                    mode = match.group(1)

                    if mode not in valid_modes:
                        line_num = content[: match.start()].count("\n") + 1
                        relative_path = str(py_file.relative_to(base_path))
                        invalid_modes.append(f"{relative_path}:{line_num} - '{mode}'")

        if invalid_modes:
            pytest.fail(
                f"Found invalid training_modes (valid: {valid_modes}):\n"
                + "\n".join(f"  - {issue}" for issue in invalid_modes[:10])
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
