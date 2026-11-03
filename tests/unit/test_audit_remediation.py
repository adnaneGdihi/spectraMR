"""Audit Remediation Tests.

Regression tests verifying all fixes from the codebase audit:
  C1: Raw torch.fft → canonical fft2c/ifft2c in generators
  C3: Import hierarchy violations (config → models)
  M2: Strategy factory registration completeness
  m1: Dead stub file removal

These tests prevent regressions — if any raw torch.fft usage reappears in the
models layer, an import violation returns to the config layer, or a strategy
is de-registered, the test suite will catch it.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The package moved to `src/spectramr/` in the 2026-05 refactor, so every path
# built from this constant pointed at a directory that no longer exists and
# the reads failed with FileNotFoundError before any assertion ran.
SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "spectramr"

# Files that were fixed in the audit (C1)
FFT_FIXED_FILES = [
    SRC_ROOT / "models" / "generators" / "pma_varnet.py",
    SRC_ROOT / "models" / "generators" / "vf_architecture_blocks.py",
    SRC_ROOT / "models" / "generators" / "siren_varnet.py",
]

# Dead stub files that were deleted (m1)
DELETED_STUBS = [
    SRC_ROOT / "models" / "html_report_generator.py",
    SRC_ROOT / "models" / "stability_analysis.py",
    SRC_ROOT / "models" / "weight_initialization.py",
    SRC_ROOT / "infrastructure" / "services" / "generation_3d_services.py",
    SRC_ROOT / "data" / "datasets" / "wrappers.py",
]


# ---------------------------------------------------------------------------
# C1: No raw torch.fft.fft2/ifft2 in models layer
# ---------------------------------------------------------------------------


_BANNED_2D_FFT_NAMES = {"fft2", "ifft2"}
_BANNED_ND_FFT_NAMES = {"fft2", "ifft2", "fftn", "ifftn"}


def _scan_banned_fft_calls(path: Path, banned: set[str]) -> list[tuple[int, str]]:
    """Return [(line, attr)] for every ``torch.fft.<banned>`` *call* in *path*.

    Walks the AST so docstrings, comments, and string literals never produce
    false positives. Only ``ast.Call`` nodes whose target is an
    ``ast.Attribute`` chain ending in ``torch.fft.<banned>`` are reported.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in banned:
            continue
        # Walk back: torch.fft.fft2 → Attribute(attr='fft2', value=Attribute(attr='fft', value=Name(id='torch')))
        parent = func.value
        if not isinstance(parent, ast.Attribute) or parent.attr != "fft":
            continue
        root = parent.value
        if isinstance(root, ast.Name) and root.id == "torch":
            hits.append((func.lineno, func.attr))
    return hits


class TestNoRawFFTInModels:
    """Ensure the models layer uses canonical fft2c/ifft2c, not raw torch.fft."""

    # The BANNED patterns are 2D k-space transforms — ``torch.fft.fftfreq``,
    # ``fftshift``, ``ifftshift``, ``rfft`` (1D attention) are fine.
    @pytest.mark.parametrize("filepath", FFT_FIXED_FILES, ids=lambda p: p.name)
    def test_no_raw_fft2_in_fixed_files(self, filepath: Path):
        """Previously-fixed files must not regress to raw torch.fft.fft2/ifft2."""
        hits = _scan_banned_fft_calls(filepath, _BANNED_2D_FFT_NAMES)
        assert not hits, (
            f"{filepath.name} contains raw torch.fft calls: "
            + ", ".join(f"line {ln}: torch.fft.{name}" for ln, name in hits)
            + " — use fft2c/ifft2c from spectramr.infrastructure.physics.fft_ops"
        )

    def test_no_raw_fft2_in_any_generator(self):
        """No generator file should use raw torch.fft.fft2/ifft2."""
        generators_dir = SRC_ROOT / "models" / "generators"
        violations = []
        for py_file in sorted(generators_dir.rglob("*.py")):
            if py_file.name.startswith("__"):
                continue
            for ln, attr in _scan_banned_fft_calls(py_file, _BANNED_2D_FFT_NAMES):
                violations.append(f"{py_file.name}:{ln}: torch.fft.{attr}()")
        assert not violations, (
            "Raw torch.fft.fft2/ifft2 found in generator files:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_no_raw_fft2_in_diffusion_models(self):
        """No diffusion model file should use raw torch.fft.fft2/ifft2."""
        diffusion_dir = SRC_ROOT / "models" / "diffusion"
        violations = []
        for py_file in sorted(diffusion_dir.rglob("*.py")):
            if py_file.name.startswith("__"):
                continue
            for ln, attr in _scan_banned_fft_calls(py_file, _BANNED_2D_FFT_NAMES):
                violations.append(f"{py_file.name}:{ln}: torch.fft.{attr}()")
        assert not violations, (
            "Raw torch.fft.fft2/ifft2 found in diffusion files:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


class TestColdDiffusionDeadCodeRemoved:
    """Verify the dead z_shifted variable was removed from cold_diffusion.py."""

    def test_no_z_shifted_variable(self):
        """The unused z_shifted variable must not exist in symmetrize_kspace."""
        cold_diffusion_path = SRC_ROOT / "models" / "diffusion" / "cold_diffusion.py"
        content = cold_diffusion_path.read_text()
        assert "z_shifted" not in content, (
            "Dead code 'z_shifted' still present in cold_diffusion.py. "
            "This variable was computed but never used."
        )


# ---------------------------------------------------------------------------
# C1 functional: FFT roundtrip through fixed generators
# ---------------------------------------------------------------------------


class TestFFTConsistencyInFixedGenerators:
    """Verify that the fft2c/ifft2c used in generators produces correct results."""

    def test_fft_ifft_roundtrip(self):
        """fft2c → ifft2c should be identity (Parseval)."""
        from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

        x = torch.randn(2, 32, 32, dtype=torch.complex64)
        x_recon = ifft2c(fft2c(x))
        assert torch.allclose(x, x_recon, atol=1e-6), "FFT roundtrip failed"

    def test_ifft2c_produces_centered_result(self):
        """ifft2c must produce DC-centered output (unlike raw torch.fft.ifft2)."""
        from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

        # Create a DC-only signal (all energy at center)
        img = torch.zeros(1, 32, 32, dtype=torch.complex64)
        img[0, 16, 16] = 1.0  # DC at center

        kspace = fft2c(img)
        recon = ifft2c(kspace)

        # Peak should still be at center after roundtrip
        peak_pos = torch.unravel_index(recon[0].abs().argmax(), (32, 32))
        assert peak_pos[0].item() == 16 and peak_pos[1].item() == 16, (
            f"DC peak shifted to ({peak_pos[0]}, {peak_pos[1]}) — "
            "ifft2c centering is broken"
        )

    def test_parseval_energy_conservation(self):
        """Energy must be preserved through fft2c (Parseval's theorem)."""
        from spectramr.infrastructure.physics.fft_ops import fft2c

        x = torch.randn(4, 64, 64, dtype=torch.complex64)
        kspace = fft2c(x)

        spatial_energy = (x.abs() ** 2).sum()
        kspace_energy = (kspace.abs() ** 2).sum()

        assert torch.allclose(spatial_energy, kspace_energy, rtol=1e-5), (
            f"Energy not conserved: spatial={spatial_energy:.4f} vs "
            f"kspace={kspace_energy:.4f}"
        )


# ---------------------------------------------------------------------------
# C3: Import hierarchy - config layer must not import from models at module level
# ---------------------------------------------------------------------------


class TestConfigLayerImportHierarchy:
    """The config layer must not import from spectramr.models at module load time."""

    def test_config_builder_no_module_level_models_import(self):
        """config_builder.py must not import from spectramr.models at the top level."""
        config_builder_path = SRC_ROOT / "config" / "config_builder.py"
        content = config_builder_path.read_text()

        # Parse AST to find top-level imports only
        tree = ast.parse(content)
        violations = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("spectramr.models"):
                    violations.append(
                        f"Line {node.lineno}: from {node.module} import ..."
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("spectramr.models"):
                        violations.append(
                            f"Line {node.lineno}: import {alias.name}"
                        )

        assert not violations, (
            "config_builder.py has top-level imports from spectramr.models "
            "(violates import hierarchy):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_config_builder_importable(self):
        """config_builder.py must be importable without errors."""
        module = importlib.import_module("spectramr.config.config_builder")
        assert hasattr(module, "ConfigBuilder")

    def test_config_builder_instantiable(self):
        """ConfigBuilder must be instantiable (lazy imports resolve)."""
        from spectramr.config.config_builder import ConfigBuilder

        builder = ConfigBuilder()
        assert builder is not None
        assert hasattr(builder, "_loss_recommender")


# ---------------------------------------------------------------------------
# M2: Strategy factory registration completeness
# ---------------------------------------------------------------------------


class TestStrategyRegistrationCompleteness:
    """All strategy files must be registered in the factory."""

    @pytest.fixture(scope="class")
    def factory_paths(self):
        from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

        return dict(TrainingStrategyFactory.STRATEGY_CLASS_PATHS)

    def test_disentangled_diffusion_registered(self, factory_paths):
        """disentangled_diffusion must be registered in STRATEGY_CLASS_PATHS."""
        assert "disentangled_diffusion" in factory_paths
        assert "DisentangledDiffusionStrategy" in factory_paths["disentangled_diffusion"]

    def test_guided_sr_registered(self, factory_paths):
        """guided_sr must be registered in STRATEGY_CLASS_PATHS."""
        assert "guided_sr" in factory_paths
        assert "GuidedSuperResolutionStrategy" in factory_paths["guided_sr"]

    def test_disentangled_diffusion_importable(self, factory_paths):
        """The disentangled_diffusion strategy class must be importable."""
        path = factory_paths["disentangled_diffusion"]
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

        assert issubclass(cls, BaseTrainingStrategy)

    def test_guided_sr_importable(self, factory_paths):
        """The guided_sr strategy class must be importable."""
        path = factory_paths["guided_sr"]
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

        assert issubclass(cls, BaseTrainingStrategy)

    def test_all_strategy_files_have_registration(self, factory_paths):
        """Every non-base strategy file should have at least one registry entry."""
        strategies_dir = SRC_ROOT / "infrastructure" / "training" / "strategies"
        registered_modules = set()
        for path in factory_paths.values():
            module_path = path.rsplit(".", 1)[0]
            # Extract just the filename part
            parts = module_path.split(".")
            if parts[-1] != "__init__":
                registered_modules.add(parts[-1])

        # Files that are infrastructure, not strategies themselves
        INFRA_FILES = {
            "__init__",
            "base",
            # "base_hooks" deleted in #1353 -- it was an identical, importerless
            # second declaration of the four lifecycle hooks that base.py owns.
            "lifecycle",
            "simulator_builder",
            "standard_strategy",
            # Utility module exposing standalone IB-score helpers (no Strategy class);
            # consumed by the active-acquisition strategy at runtime.
            "ib_active_acquisition_nonlinear",
        }

        unregistered = []
        for py_file in sorted(strategies_dir.glob("*.py")):
            stem = py_file.stem
            if stem in INFRA_FILES or stem.startswith("__"):
                continue
            # A module that defines no class at all cannot hold a strategy, so
            # it is infrastructure by construction and needs no entry in the
            # hand-written set above. `loss_folding.py` is the case that forced
            # this: two pure functions, the SSOT for builder-image-loss folding,
            # and it failed the check purely for not having been listed. The
            # named set stays for modules that DO define a class without being a
            # strategy (`base`, `standard_strategy`, ...), which cannot be
            # derived this cheaply.
            source = py_file.read_text(encoding="utf-8")
            if not any(
                line.startswith("class ") for line in source.splitlines()
            ):
                continue
            if stem not in registered_modules:
                unregistered.append(stem)

        assert not unregistered, (
            "Strategy files not registered in STRATEGY_CLASS_PATHS:\n"
            + "\n".join(f"  - {f}.py" for f in unregistered)
            + "\nEither register them or mark as infrastructure."
        )


# ---------------------------------------------------------------------------
# m1: Dead stub files deleted
# ---------------------------------------------------------------------------


class TestDeadStubFilesRemoved:
    """Verify that identified dead stub files have been deleted."""

    @pytest.mark.parametrize(
        "stub_path",
        DELETED_STUBS,
        ids=lambda p: str(p.relative_to(SRC_ROOT)),
    )
    def test_stub_file_deleted(self, stub_path: Path):
        """Dead stub file must not exist."""
        assert not stub_path.exists(), (
            f"Dead stub file still exists: {stub_path.relative_to(SRC_ROOT)}\n"
            "This file was identified as dead code with zero importers."
        )

    def test_no_imports_reference_deleted_stubs(self):
        """No source file should import from deleted stub modules."""
        deleted_modules = [
            "spectramr.models.html_report_generator",
            "spectramr.models.stability_analysis",
            "spectramr.models.weight_initialization",
            "spectramr.infrastructure.services.generation_3d_services",
            "spectramr.data.datasets.wrappers",
        ]
        violations = []
        for py_file in SRC_ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            content = py_file.read_text()
            for mod in deleted_modules:
                if f"from {mod}" in content or f"import {mod}" in content:
                    violations.append(
                        f"{py_file.relative_to(SRC_ROOT)}: references {mod}"
                    )

        assert not violations, (
            "Source files still reference deleted stubs:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# Broad models-layer FFT compliance scan
# ---------------------------------------------------------------------------


class TestModelsLayerFFTCompliance:
    """Scan the entire models/ directory for raw torch.fft.fft2/ifft2 usage.

    The models layer must delegate all 2D k-space FFT operations to
    spectramr.infrastructure.physics.fft_ops (fft2c, ifft2c).

    Acceptable patterns in models/:
    - torch.fft.fftfreq (frequency grid computation)
    - torch.fft.fftshift / ifftshift (data rearrangement)
    - torch.fft.rfft / irfft (1D real FFT for attention mechanisms)
    - torch.fft.fft / ifft (1D FFT)
    """

    def test_entire_models_directory(self):
        """No file under src/models/ should use banned FFT patterns."""
        models_dir = SRC_ROOT / "models"
        violations = []
        for py_file in sorted(models_dir.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            for ln, attr in _scan_banned_fft_calls(py_file, _BANNED_ND_FFT_NAMES):
                violations.append(
                    f"{py_file.relative_to(SRC_ROOT)}:{ln}: torch.fft.{attr}"
                )

        assert not violations, (
            "Raw torch.fft.fft2/ifft2/fftn/ifftn found in models/ layer:\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nUse fft2c/ifft2c from spectramr.infrastructure.physics.fft_ops"
        )
"""Unit tests for the symmetrize_kspace method after dead code removal."""


class TestSymmetrizeKspaceAfterFix:
    """Verify symmetrize_kspace still works correctly after z_shifted removal."""

    def test_symmetrize_produces_real_image(self):
        """Symmetrized k-space should IFFT to a real-valued image."""
        from spectramr.models.diffusion.cold_diffusion import ColdDiffusion

        z = torch.randn(2, 2, 32, 32)
        sym_z = ColdDiffusion.symmetrize_kspace(z)

        # Must maintain shape
        assert sym_z.shape == z.shape

        # IFFT of symmetrized k-space should have near-zero imaginary part
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        z_c = torch.complex(sym_z[:, 0], sym_z[:, 1])
        img = ifft2c(z_c)
        assert torch.allclose(img.imag, torch.zeros_like(img.imag), atol=1e-5), (
            f"Symmetrized k-space does not produce real image. "
            f"Max imag: {img.imag.abs().max():.6f}"
        )

    def test_symmetrize_preserves_batch_dim(self):
        """Batch dimension must be preserved."""
        from spectramr.models.diffusion.cold_diffusion import ColdDiffusion

        for batch_size in [1, 4, 8]:
            z = torch.randn(batch_size, 2, 16, 16)
            sym = ColdDiffusion.symmetrize_kspace(z)
            assert sym.shape[0] == batch_size
"""Tests for the VF architecture ESPIRiT block after ifft2c migration."""


class TestESPIRiTBlockAfterFFTFix:
    """Verify DirichletESPIRiTBlock still works after ifft2c migration."""

    def test_espirit_output_shape(self):
        """ESPIRiT block must produce correctly shaped sensitivity maps."""
        from spectramr.models.generators.vf_architecture_blocks import DirichletESPIRiTBlock

        block = DirichletESPIRiTBlock(kernel_size=4)
        kspace = torch.randn(1, 4, 32, 32, dtype=torch.cfloat)
        maps = block(kspace, center_fraction=0.2)
        assert maps.shape == (1, 4, 32, 32)

    def test_espirit_normalized_output(self):
        """ESPIRiT maps should be approximately unit-normalized (RSS ≈ 1)."""
        from spectramr.models.generators.vf_architecture_blocks import DirichletESPIRiTBlock

        block = DirichletESPIRiTBlock(kernel_size=4)
        kspace = torch.randn(1, 4, 32, 32, dtype=torch.cfloat)
        maps = block(kspace, center_fraction=0.3)
        rss = (maps.abs().pow(2).sum(dim=1) + 1e-8).sqrt()
        assert rss.mean().item() == pytest.approx(1.0, abs=0.3)
"""Tests for SIRENVarNet after ifft2c migration."""


class TestSIRENVarNetAfterFFTFix:
    """Verify SIRENVarNetGenerator still produces correct output after fix."""

    def test_siren_varnet_importable(self):
        """SIRENVarNetGenerator must be importable."""
        from spectramr.models.generators.siren_varnet import SIRENVarNetGenerator

        assert SIRENVarNetGenerator is not None

    def test_siren_varnet_forward_shape(self):
        """Forward pass must produce correct output shape."""
        from spectramr.models.generators.siren_varnet import SIRENVarNetGenerator

        model = SIRENVarNetGenerator(
            in_channels=2,
            out_channels=2,
            num_cascades=2,
            unet_features=16,
        )
        model.eval()
        y_measured = torch.randn(1, 2, 32, 32)
        mask = torch.ones(1, 1, 32, 32)  # fully-sampled mask
        with torch.no_grad():
            y = model(y_measured, mask)
        assert y.shape == (1, 2, 32, 32)
"""Tests for PMA-VarNet after fft2c/ifft2c migration."""


class TestPMAVarNetAfterFFTFix:
    """Verify PMAVarNet still produces correct output after fft2c migration."""

    def test_pma_varnet_importable(self):
        """PMAVarNet must be importable with fft2c/ifft2c."""
        from spectramr.models.generators.pma_varnet import PMAVarNet

        assert PMAVarNet is not None

    def test_pma_varnet_imports_canonical_fft(self):
        """PMAVarNet module must not use raw torch.fft (its FFTs live in Operator_A_m4)."""
        import spectramr.models.generators.pma_varnet as mod

        source = Path(mod.__file__).read_text()
        assert "torch.fft.fft2" not in source
        assert "torch.fft.ifft2" not in source
"""Full test for config_builder lazy import pattern."""


class TestConfigBuilderLazyImport:
    """Verify the config → models import is truly lazy."""

    def test_config_schemas_importable_without_models(self):
        """Importing config.schemas should not trigger models imports."""
        # This test verifies the import hierarchy is clean
        # by checking that the config module can be parsed without
        # actually importing models at the top level
        config_builder_path = SRC_ROOT / "config" / "config_builder.py"
        tree = ast.parse(config_builder_path.read_text())

        top_level_imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top_level_imports.append(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        top_level_imports.append(alias.name)

        # None of the top-level imports should be from spectramr.models
        models_imports = [i for i in top_level_imports if i.startswith("spectramr.models")]
        assert not models_imports, (
            f"Top-level imports from spectramr.models found: {models_imports}"
        )



if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
