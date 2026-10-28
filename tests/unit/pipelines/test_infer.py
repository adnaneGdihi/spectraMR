"""
Unit tests for pipelines/infer.py - Inference pipeline.

Tests the inference pipeline for model predictions.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml as _yaml

from spectramr.config.schemas.logging import LoggingConfigSchema
from spectramr.domain.exceptions import (
    ConfigurationError,
    DataCorruptionError,
    DimensionMismatchError,
)


class MockModel(nn.Module):
    """Simple mock model for testing."""

    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestInferencePipeline:
    """Tests for the inference pipeline."""

    def test_model_inference_basic(self):
        """Test basic model inference."""
        model = MockModel(2, 2)
        x = torch.randn(1, 2, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == x.shape

    def test_inference_with_eval_mode(self):
        """Test that inference uses eval mode."""
        model = MockModel(2, 2)
        model.eval()

        assert not model.training

    def test_inference_no_grad(self):
        """Test inference without gradient computation."""
        model = MockModel(2, 2)
        x = torch.randn(1, 2, 64, 64, requires_grad=True)

        with torch.no_grad():
            output = model(x)

        # Output should not require grad in no_grad context
        assert not output.requires_grad

    def test_batch_inference(self):
        """Test inference with batched input."""
        model = MockModel(2, 2)
        model.eval()

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 2, 64, 64)
            with torch.no_grad():
                output = model(x)
            assert output.shape[0] == batch_size

    def test_inference_deterministic(self):
        """Test that inference is deterministic."""
        model = MockModel(2, 2)
        model.eval()
        x = torch.randn(1, 2, 32, 32)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)


class TestSlidingWindowInference:
    """Tests for sliding window inference pattern."""

    def test_sliding_window_basic(self):
        """Test basic sliding window concept."""
        # Simulate sliding window on a large image
        full_image = torch.randn(1, 2, 256, 256)
        patch_size = 64
        stride = 32

        patches = []
        for i in range(0, full_image.shape[2] - patch_size + 1, stride):
            for j in range(0, full_image.shape[3] - patch_size + 1, stride):
                patch = full_image[:, :, i : i + patch_size, j : j + patch_size]
                patches.append(patch)

        assert len(patches) > 0
        assert all(p.shape == (1, 2, patch_size, patch_size) for p in patches)

    def test_overlapping_patches(self):
        """Test that overlapping patches cover the entire image."""
        H, W = 128, 128
        patch_size = 64
        stride = 32

        coverage = torch.zeros(H, W)
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                coverage[i : i + patch_size, j : j + patch_size] += 1

        # Interior should be covered multiple times
        assert coverage[patch_size // 2, patch_size // 2] > 1


class TestTestTimeAugmentation:
    """Tests for Test Time Augmentation (TTA) pattern."""

    def test_tta_flips(self):
        """Test TTA with horizontal and vertical flips."""
        model = MockModel(2, 2)
        model.eval()
        x = torch.randn(1, 2, 64, 64)

        # Original
        with torch.no_grad():
            pred_orig = model(x)

        # Horizontal flip
        x_hflip = torch.flip(x, dims=[-1])
        with torch.no_grad():
            pred_hflip = model(x_hflip)
        pred_hflip = torch.flip(pred_hflip, dims=[-1])

        # Vertical flip
        x_vflip = torch.flip(x, dims=[-2])
        with torch.no_grad():
            pred_vflip = model(x_vflip)
        pred_vflip = torch.flip(pred_vflip, dims=[-2])

        # Average predictions
        pred_avg = (pred_orig + pred_hflip + pred_vflip) / 3

        assert pred_avg.shape == x.shape


class TestLoadInputRoutesThroughSSOT:
    """Review 2026-07-01: ``_load_input`` must go through the data-layer SSOT
    (``io_strategies.load_tensor_from_file``), not call ``h5py.File`` / ``nib.load``
    / ``np.load`` directly from the pipeline layer (CLAUDE.md pitfall #11)."""

    def test_h5_key_precedence_via_ssot(self, tmp_path):
        import pytest

        h5py = pytest.importorskip("h5py")
        import numpy as np

        from spectramr.pipelines.infer import _load_input

        p = tmp_path / "scan.h5"
        ksp = np.arange(6, dtype=np.float32).reshape(2, 3)
        rec = np.zeros((2, 3), dtype=np.float32)
        with h5py.File(p, "w") as f:
            f.create_dataset("reconstruction_rss", data=rec)
            f.create_dataset("kspace", data=ksp)

        out = _load_input(p, torch.device("cpu"))
        # 'kspace' is preferred over 'reconstruction_rss' → proves the SSOT
        # loader's h5_keys precedence is being used.
        assert torch.equal(out, torch.from_numpy(ksp).float())

    def test_load_input_has_no_inline_readers(self):
        """Source guard: the pipeline function must not re-inline file readers."""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[3] / "src" / "spectramr" / "pipelines" / "infer.py"
        ).read_text()
        start = src.index("def _load_input(")
        end = src.index("\ndef ", start + 1)
        body = src[start:end]
        for needle in ("h5py.File(", "nib.load(", "np.load("):
            assert needle not in body, f"{needle} re-inlined in _load_input"


class TestResolveInferenceParadigm:
    """WS-A A2: paradigm resolution must defer to the SSOT strategy detector
    (the one the factory dispatches on), never record a divergent ``"unknown"``.

    The load-bearing exception is the multi-stage gate: the detector has no
    ``"multi"`` rule, so a ``strategy_class`` name containing ``multi`` must
    short-circuit to ``"multi"`` WITHOUT consulting the detector.
    """

    @staticmethod
    def _cfg(strategy_class):
        from types import SimpleNamespace

        return SimpleNamespace(
            training=SimpleNamespace(strategy_class=strategy_class),
            model_dump=lambda: {"training": {"strategy_class": strategy_class}},
        )

    def test_multi_stage_short_circuits_without_detector(self, monkeypatch):
        from spectramr.pipelines import infer

        # If the detector is consulted for a multi-stage class, fail — the gate
        # must short-circuit (detector cannot emit "multi").
        monkeypatch.setattr(
            "spectramr.infrastructure.inference.inference_factory."
            "InferenceStrategyFactory._infer_strategy_type",
            lambda cfg: (_ for _ in ()).throw(
                AssertionError("detector must not be called for multi-stage")
            ),
        )
        cfg = self._cfg(
            "spectramr.infrastructure.training.strategies.pipeline_strategy.MultiTrainingStrategy"
        )
        assert infer._resolve_inference_paradigm(cfg) == "multi"

    def test_non_multi_defers_to_ssot_detector(self, monkeypatch):
        from spectramr.pipelines import infer

        monkeypatch.setattr(
            "spectramr.infrastructure.inference.inference_factory."
            "InferenceStrategyFactory._infer_strategy_type",
            lambda cfg: "gan",
        )
        cfg = self._cfg("spectramr...GANTrainingStrategy")
        # The SSOT detector's verdict is used verbatim — never a stale "unknown".
        result = infer._resolve_inference_paradigm(cfg)
        assert result == "gan"
        assert result != "unknown"

    def test_absent_strategy_class_still_defers_to_detector(self, monkeypatch):
        from types import SimpleNamespace

        from spectramr.pipelines import infer

        monkeypatch.setattr(
            "spectramr.infrastructure.inference.inference_factory."
            "InferenceStrategyFactory._infer_strategy_type",
            lambda cfg: "reconstruction",
        )
        cfg = SimpleNamespace(
            training=SimpleNamespace(strategy_class=None),
            model_dump=lambda: {},
        )
        assert infer._resolve_inference_paradigm(cfg) == "reconstruction"


class TestInferenceAcceleratedRunContract:
    """Inference is a heavy pipeline: no accelerator → raise, never a CPU run.

    ``run_inference_pipeline`` used to do a bare ``torch.device(device)``, which
    neither honoured ``"auto"`` (``torch.device("auto")`` is a TypeError) nor
    checked CUDA availability — it built a cuda device object on a GPU-less host
    and only failed later, deep inside a ``.to(device)``. It now routes through
    the SSOT contract in :mod:`spectramr.core.compute_device`.
    """

    def test_pipeline_resolves_device_through_the_ssot(self) -> None:
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[3] / "src" / "spectramr" / "pipelines" / "infer.py"
        ).read_text()
        assert "resolve_torch_device" in src
        assert "device_obj = torch.device(device)" not in src, (
            "infer.py still builds the device without the accelerated-run "
            "contract; 'auto' would crash and a GPU-less host would degrade."
        )

    def test_no_accelerator_raises_for_infer(self, monkeypatch) -> None:
        import torch

        from spectramr.core.compute_device import (
            AcceleratorRequiredError,
            resolve_torch_device,
        )

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.delenv("FORCE_CPU", raising=False)

        with pytest.raises(AcceleratorRequiredError):
            resolve_torch_device("cuda", pipeline="infer")

    def test_explicit_cpu_inference_is_permitted(self, monkeypatch) -> None:
        import torch

        from spectramr.core.compute_device import resolve_torch_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        decision = resolve_torch_device("cpu", pipeline="infer")
        assert decision.device == "cpu"
        assert decision.cpu_opt_in is True


def test_infer_preprocess_gates_on_declared_normalization_fields() -> None:
    """The preprocessing gate read `config.data.normalize` and then
    `config.data.normalization_percentile`; neither is declared, so it raised on
    the gate before reaching the value. The canonical knobs are
    `normalization_type` (a closed Literal naming both percentile variants).

    NOTE: this asserts schema shape only. `rescale_percentiles` is checked here
    for validity, NOT because inference reads it -- it no longer does, and no
    training code ever did (see
    ``TestInferenceNormalizesExactlyAsTrainingDoes``)."""
    from spectramr.config.schemas.data import DataConfigSchema

    cfg = DataConfigSchema()
    assert cfg.processing.normalization_type in {
        "none",
        "standard",
        "minmax",
        "percentile",
        "robust_percentile",
        "scalar",
    }
    lo, hi = cfg.processing.rescale_percentiles
    assert 0.0 <= lo <= hi <= 100.0
    # The two literals the gate now tests for really are in the Literal set.
    for variant in ("percentile", "robust_percentile"):
        assert DataConfigSchema(normalization_type=variant).processing.normalization_type == variant


class TestManifestTestSplitRoute:
    """``from_manifest_test_split`` selects the held-out cohort as the roster.

    The capability came from ``DataPipelineDirector._resolve_manifest_test_paths``,
    whose only caller (``build_inference_handle``) was deleted as a duplicate of
    the parity check this pipeline already performs. The manifest route had no
    duplicate here -- ``_collect_input_files`` globs the filesystem -- so it was
    rehomed rather than dropped.
    """

    def test_pipeline_accepts_the_flag(self) -> None:
        import inspect

        from spectramr.pipelines.infer import run_inference_pipeline

        params = inspect.signature(run_inference_pipeline).parameters
        assert "from_manifest_test_split" in params
        assert params["from_manifest_test_split"].default is False, (
            "the manifest route must be opt-in: defaulting it on would switch "
            "every existing inference run's data source"
        )
        # input_path must be optional-valued, since the manifest route supplies
        # the roster instead.
        import typing

        assert type(None) in typing.get_args(params["input_path"].annotation), (
            "input_path must accept None so --from-manifest-test-split can "
            "supply the roster without a dummy path"
        )

    def test_route_is_explicit_not_an_empty_glob_fallback(self) -> None:
        """A mistyped --input must raise, never silently switch sources."""
        import inspect

        from spectramr.pipelines import infer as infer_mod

        src = inspect.getsource(infer_mod.run_inference_pipeline)
        code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
        assert "if from_manifest_test_split:" in code
        assert "resolve_manifest_test_paths(" in code
        # The glob branch must still raise on an empty result rather than
        # falling through to the manifest.
        assert "No input files found in" in code

    def test_cli_parses_the_flag(self) -> None:
        """The knob is declared on the CLI (non-negotiable #8)."""
        from spectramr.cli.app import build_parser

        args = build_parser().parse_args(
            [
                "infer",
                "--config",
                "c.yaml",
                "--checkpoint",
                "m.pt",
                "--from-manifest-test-split",
            ]
        )
        assert args.from_manifest_test_split is True
        # --input is no longer required, since the manifest supplies the roster.
        assert getattr(args, "input", None) is None

    def test_the_handler_the_parser_binds_forwards_the_flag(self) -> None:
        """Parsing it is not wiring it.

        The `infer` subcommand binds ``cli.app.infer``, which delegates to
        ``main.infer_command`` -- NOT to ``cli.app.predict``. A first cut of
        this change added the pass-through to ``predict``, so the flag parsed
        fine and reached nothing: a declared-but-unread knob (pitfall #15) in
        the very change meant to wire one. Follow the binding, not the name.
        """
        import inspect

        from spectramr.cli.app import build_parser

        args = build_parser().parse_args(["infer", "--config", "c.yaml", "--checkpoint", "m.pt"])
        bound = args.func  # what the parser actually dispatches to

        chain = inspect.getsource(bound)
        assert "infer_command" in chain, (
            f"the infer subcommand now binds {bound.__name__}; re-check which "
            "handler must forward from_manifest_test_split"
        )

        from spectramr.main import infer_command

        forwarded = inspect.getsource(infer_command)
        code = "\n".join(
            line for line in forwarded.splitlines() if not line.lstrip().startswith("#")
        )
        assert "from_manifest_test_split" in code, (
            "main.infer_command does not forward from_manifest_test_split, so the CLI flag is inert"
        )


class TestInferBuildsThroughTheCanonicalBuilder:
    """Inference must construct models the way training does.

    ``infer`` used to call ``ModelFactory.create_model(config.model)``. Handed a
    bare ``ModelConfigSchema``, that selected a branch resolving a strict subset
    of the builder's config->kwargs work: no ``acceleration_config``, no
    ``kspace_log_scaled``, none of the ``ModelConfigSchema`` sweep. The model was
    built differently from the one the same YAML trains, and nothing said so.

    ``ModelBuilder`` is imported function-locally in ``infer.py`` (torch import
    cost), so the module-level binding these tests can see is
    ``weights``; the builder itself is pinned by source guard, the
    idiom this module already uses for ``_load_input`` and the device contract.
    """

    @staticmethod
    def _infer_src() -> str:
        from pathlib import Path

        return (
            Path(__file__).resolve().parents[3] / "src" / "spectramr" / "pipelines" / "infer.py"
        ).read_text()

    def test_pipeline_no_longer_calls_the_config_sniffing_factory(self) -> None:
        """AST, not grep: the retired call must be gone from the *code*.

        A text search also matches the comment that explains what was replaced,
        so it would force the explanation out of the file to stay green. Parsing
        asks the question actually being asked -- is there a call to it?
        """
        import ast

        tree = ast.parse(self._infer_src())
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute | ast.Name)
        }
        assert "create_model" not in called, (
            "infer.py still routes through ModelFactory.create_model, which "
            "resolves a strict subset of the builder's kwargs"
        )
        assert "get_model_factory" not in called

        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "get_model_factory" not in imported

    def test_pipeline_builds_through_model_builder(self) -> None:
        assert "ModelBuilder(config, device_obj)" in self._infer_src()

    def test_pipeline_binds_the_one_weight_reader(self) -> None:
        import spectramr.pipelines.infer as infer_mod

        assert infer_mod.weights.__module__ == "spectramr.infrastructure.checkpointing.restore"

    @staticmethod
    def _weights_calls(src: str) -> list[tuple[object, object]]:
        """``(part, policy)`` of every ``weights(...)`` call, read off the AST."""
        import ast

        calls = []
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "weights":
                part = node.args[1].value if len(node.args) > 1 else None
                policy = next((k.value.value for k in node.keywords if k.arg == "policy"), None)
                calls.append((part, policy))
        return calls

    def test_pipeline_loads_the_generator_part_exactly(self) -> None:
        """Predict scores the trained generator, all of it: neither the EMA shadow nor a partial load."""
        assert self._weights_calls(self._infer_src()) == [("generator", "exact")]


class TestAccelerationConfigReachesTheModel:
    """The silent half of #1306, pinned on the arm that exhibits it.

    #1306 is the loud symptom -- a hard raise on the 12 ``kspace_filling`` arms
    that set ``output_kspace_clip_ratio``. The quiet symptom reaches all 58 arms
    in the cohort: every one declares ``undersampling:`` and every one is
    ``kspace_cold_diffusion``, one of exactly two generators whose ``__init__``
    accepts ``acceleration_config``. Without it the constructor falls back to
    ``resolve_undersampling_kwargs({}, {})``, so inference reconstructs with a
    **different ACS band than training used** -- and reports success.

    The numbers come from the arm, never from class defaults: the two assertions
    below are a sensitivity pair, so the declared value cannot silently coincide
    with the fallback and make the test vacuous.
    """

    ARM = (
        "experiments/inprogress/kspace_filling/attention_shootout/experiment_11_attention_none.yaml"
    )

    def _config(self):
        from pathlib import Path

        import pytest as _pytest

        from spectramr.config.settings import TrainingSettings

        arm = Path(__file__).resolve().parents[3] / self.ARM
        if not arm.is_file():
            _pytest.skip(f"cohort arm not present at {self.ARM}")
        return TrainingSettings.from_yaml(str(arm))

    def test_builder_injects_the_declared_acs_band_not_the_default(self) -> None:
        import torch

        from spectramr.infrastructure.training.builders.model_builder import (
            ModelBuilder,
        )
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        config = self._config()
        declared = config.undersampling.center_fraction
        fallback = resolve_undersampling_kwargs({}, {})["center_fraction"]

        # Sensitivity guard: if these ever coincide the assertion below stops
        # discriminating and must be re-anchored on a different arm.
        assert declared != fallback, (
            f"arm's center_fraction ({declared}) equals the no-config fallback "
            f"({fallback}); this test can no longer detect the drop"
        )

        generator = (
            ModelBuilder(config, torch.device("cpu"))
            .build_generator()
            .validate()
            .build()["generator"]
        )
        assert generator.center_fraction == declared, (
            f"acceleration_config did not reach the model: got "
            f"{generator.center_fraction}, arm declares {declared}"
        )

    def test_builder_injects_kspace_log_scaled_from_the_processing_ssot(self) -> None:
        """The #1306 raise itself: set clip ratio + absent flag == ValueError."""
        import torch

        from spectramr.infrastructure.training.builders.model_builder import (
            ModelBuilder,
        )

        config = self._config()
        generator = (
            ModelBuilder(config, torch.device("cpu"))
            .build_generator()
            .validate()
            .build()["generator"]
        )
        assert generator._output_kspace_clip_ratio is not None, (
            "arm no longer sets output_kspace_clip_ratio; #1306 needs it"
        )
        from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

        compressed = KSpaceNormalizationSpec.from_data_config(config.data).log_compressed
        assert generator._kspace_log_scaled is compressed, (
            "kspace_log_scaled must mirror KSpaceNormalizationSpec.log_compressed"
        )


# ---------------------------------------------------------------------------
# Train/infer preprocessing parity (PR2 item 6)
# ---------------------------------------------------------------------------


def _norm_arm(
    tmp_path: Path,
    *,
    normalization_type: str = "none",
    enable_kspace_normalization: bool = False,
    rescale_percentiles: tuple = (0.0, 100.0),
    dataset_type: str = "m4raw",
):
    """A minimal arm whose image-normalization block is fully declared."""
    from spectramr.config.settings import TrainingSettings

    cfg = {
        "config_version": "1.0",
        "model": {"model_type": "unet", "in_channels": 2, "out_channels": 2},
        "data": {
            "dataset_type": dataset_type,
            "sampling": {"patch_size": [16, 16]},
            "loader": {"batch_size": 1},
            "processing": {
                "normalization_type": normalization_type,
                "enable_kspace_normalization": enable_kspace_normalization,
                "rescale_percentiles": list(rescale_percentiles),
            },
        },
        "optimization": {},
        "logging": {},
    }
    path = tmp_path / "norm_arm.yaml"
    path.write_text(_yaml.safe_dump(cfg))
    return TrainingSettings.from_yaml(str(path))


class TestInferenceNormalizesExactlyAsTrainingDoes:
    """``_normalize_like_training`` mirrors the training transform builder.

    The divergence this closes was not a parameter mismatch -- it was whether
    the operation runs at all. ``TorchIOTransformBuilder`` gates image
    normalization on ``if not normalize_kspace``; the old inference code had no
    such gate, so across ``experiments/inprogress/kspace_filling`` (58/58 arms
    set ``enable_kspace_normalization: true``, 46 declare
    ``normalization_type: robust_percentile``) training normalized none of them
    and predict windowed 46.
    """

    def test_a_kspace_normalized_arm_gets_no_image_normalization(self, tmp_path):
        """The mutual exclusion, which is the whole cohort's situation."""
        from spectramr.pipelines.infer import _normalize_like_training

        config = _norm_arm(
            tmp_path,
            normalization_type="robust_percentile",
            enable_kspace_normalization=True,
        )
        tensor = torch.rand(1, 2, 8, 8) * 100.0
        out = _normalize_like_training(tensor.clone(), config)
        assert torch.equal(out, tensor), (
            "training SKIPS image normalization when normalize_kspace=True "
            "(preserving complex k-space); inference must skip it too"
        )

    def test_the_percentile_path_matches_the_training_transform(self, tmp_path):
        """The invariant: same tensor, same numbers, both entry points.

        Runs the real ``ImageNormalizationTransform`` -- the object the training
        chain appends -- over a ``tio.Subject`` and compares it against what
        inference produces for the same declared config.
        """
        import torchio as tio

        from spectramr.data.transforms.normalization import (
            ImageNormalizationSpec,
            ImageNormalizationTransform,
        )
        from spectramr.pipelines.infer import _normalize_like_training

        config = _norm_arm(
            tmp_path,
            normalization_type="robust_percentile",
            enable_kspace_normalization=False,
        )
        tensor = torch.rand(1, 2, 8, 8) * 100.0

        spec = ImageNormalizationSpec.from_declared(
            "percentile", config.data.dataset_type, config.data.processing.normalization_kwargs
        )
        subject = tio.Subject(input=tio.ScalarImage(tensor=tensor[0].unsqueeze(-1).clone()))
        expected = ImageNormalizationTransform(spec)(subject)["input"].data

        actual = _normalize_like_training(tensor.clone(), config)

        assert torch.allclose(actual[0].unsqueeze(-1), expected, atol=1e-6), (
            "inference and the training transform disagree on the same arm"
        )

    def test_rescale_percentiles_no_longer_steers_inference(self, tmp_path):
        """It steered inference and nothing else; no training code reads it."""
        from spectramr.pipelines.infer import _normalize_like_training

        tensor = torch.rand(1, 2, 8, 8) * 100.0
        wide = _norm_arm(
            tmp_path, normalization_type="percentile", rescale_percentiles=(0.0, 100.0)
        )
        narrow = _norm_arm(
            tmp_path, normalization_type="percentile", rescale_percentiles=(5.0, 90.0)
        )
        assert torch.allclose(
            _normalize_like_training(tensor.clone(), wide),
            _normalize_like_training(tensor.clone(), narrow),
        ), "inference must not read a field training ignores"

    def test_an_unknown_normalization_type_raises(self, tmp_path):
        """No silent degrade to 'none' (non-negotiable 3).

        Defence in depth: the schema Literal is the first gate, so this value
        cannot arrive from a validated YAML. The resolver must still refuse it
        rather than fall through, because inference resolves the string itself.
        """
        from spectramr.pipelines.infer import _normalize_like_training

        config = _norm_arm(tmp_path, normalization_type="none")
        object.__setattr__(config.data.processing, "normalization_type", "not_a_strategy")
        with pytest.raises(ConfigurationError, match="Unknown normalization_type"):
            _normalize_like_training(torch.rand(1, 2, 8, 8), config)


class TestImageNormalizationHasOneOwner:
    """Training and predict resolve the SAME image-normalization spec.

    They used to each carry a copy of the three-part decision (k-space
    precedence, the ``robust_percentile`` fold, the spec), and the copies had
    drifted once already (see :class:`TestInferenceNormalizesExactlyAsTrainingDoes`).
    The guard here is structural: both call sites go through
    ``resolve_image_normalization`` and get the same answer for the same arm,
    across the reference template and two committed arms with opposite
    k-space settings.
    """

    TEMPLATE = "src/spectramr/config/schemas/templates/v1.0_reference.yaml"
    ARMS = (
        "experiments/inprogress/kspace_filling/experiment_11_kspace_cold_diffusion.yaml",
        "experiments/inprogress/reconstruction/baseline_m4raw_unet_4x.yaml",
    )

    @staticmethod
    def _settings(rel: str):
        from spectramr.config.settings import TrainingSettings

        path = Path(__file__).resolve().parents[3] / rel
        if not path.exists():
            pytest.skip(f"not present: {rel}")
        return TrainingSettings.from_yaml(str(path))

    @staticmethod
    def _record_both(monkeypatch, config):
        """Run the builder's val chain and predict's normalizer under one spy."""
        from spectramr.data.builders.torchio_transform_builder import (
            TorchIOTransformBuilder,
            TorchIOTransformConfig,
        )
        from spectramr.data.transforms import normalization as norm_mod
        from spectramr.pipelines.infer import _normalize_like_training

        real = norm_mod.resolve_image_normalization
        seen: list[tuple[dict, object]] = []

        def _spy(**kwargs):
            spec = real(**kwargs)
            seen.append((kwargs, spec))
            return spec

        monkeypatch.setattr(norm_mod, "resolve_image_normalization", _spy)

        class _Proxy:
            # ``from_training_config`` reads ``config.undersampling`` by name;
            # the same shape ``signature.compute_infer_signature`` uses.
            def __init__(self, data_cfg, undersampling_cfg):
                self._data = data_cfg
                self.undersampling = undersampling_cfg

            def __getattr__(self, name):
                return getattr(self._data, name)

        tcfg = TorchIOTransformConfig.from_training_config(
            _Proxy(config.data, getattr(config, "undersampling", None))
        )
        TorchIOTransformBuilder.build_val_transforms(tcfg)
        assert len(seen) == 1, "the builder must resolve exactly once"
        _normalize_like_training(torch.rand(1, config.model.in_channels, 8, 8), config)
        assert len(seen) == 2, "predict must resolve exactly once"
        return seen

    @pytest.mark.parametrize("rel", (TEMPLATE, *ARMS))
    def test_builder_and_predict_resolve_the_same_spec(self, monkeypatch, rel):
        config = self._settings(rel)
        (builder_kwargs, builder_spec), (infer_kwargs, infer_spec) = self._record_both(
            monkeypatch, config
        )
        assert builder_kwargs == infer_kwargs, "the two call sites read different knobs"
        assert builder_spec == infer_spec
        processing = config.data.processing
        assert builder_kwargs["normalization_type"] == processing.normalization_type, (
            "the declared spelling must reach the resolver unfolded"
        )

    @pytest.mark.parametrize("rel", ARMS)
    def test_a_kspace_normalized_arm_resolves_to_none_on_both_sides(self, monkeypatch, rel):
        config = self._settings(rel)
        assert config.data.processing.enable_kspace_normalization is True, rel
        for _kwargs, spec in self._record_both(monkeypatch, config):
            assert spec is None

    def test_the_fold_is_invisible_to_the_transform_signature(self, tmp_path):
        """``robust_percentile`` and ``percentile`` hash to the same chain.

        ``strict_train_parity`` compares the chain signature a checkpoint
        recorded against the one predict recomputes; the signature hashes the
        transforms' classes and kwargs, so the spec, never the declared
        spelling. Pinned here because the fold moved out of the builder and a
        signature shift would refuse every strict checkpoint of the 270 arms
        that spell it ``robust_percentile``.
        """
        from spectramr.data.transforms.signature import compute_infer_signature

        alias = _norm_arm(tmp_path, normalization_type="robust_percentile")
        canonical = _norm_arm(tmp_path, normalization_type="percentile")
        none = _norm_arm(tmp_path, normalization_type="none")
        assert compute_infer_signature(alias) == compute_infer_signature(canonical)
        assert compute_infer_signature(alias) != compute_infer_signature(none), (
            "control: the signature must still see the normalizer at all"
        )

    def test_predict_no_longer_spells_the_fold(self):
        """The copy that was deleted from ``_normalize_like_training``."""
        import inspect
        import re

        from spectramr.pipelines import infer as infer_mod

        src = inspect.getsource(infer_mod._normalize_like_training)
        code_lines = [line.split("#", 1)[0] for line in src.splitlines()]
        assert not [
            line for line in code_lines if re.search(r"""==\s*['"]robust_percentile['"]""", line)
        ], "predict folds robust_percentile itself again"


class TestInferenceRefusesToReshapeTheAcquisition:
    """``_adapt_channels`` verifies; it no longer adapts.

    All three old branches changed the data and logged success at INFO: padding
    fabricated unmeasured channels, ``narrow()`` discarded measured ones, and
    the RSS branch read the channel axis as coil-major interleaved when this
    repo packs stacked halves -- while duplicating
    ``data.coils.processing_mode='rss'``, which the data pipeline has already
    applied by this point.
    """

    def test_matching_channels_pass_through(self, tmp_path):
        """Regression guard: the agreeing case must stay a no-op."""
        from spectramr.pipelines.infer import _adapt_channels

        config = _norm_arm(tmp_path)
        tensor = torch.rand(1, 2, 8, 8)
        assert torch.equal(_adapt_channels(tensor, 2, config), tensor)

    def test_too_few_channels_raises_instead_of_zero_padding(self, tmp_path):
        from spectramr.pipelines.infer import _adapt_channels

        config = _norm_arm(tmp_path)
        with pytest.raises(DimensionMismatchError, match="fabricates channels"):
            _adapt_channels(torch.rand(1, 1, 8, 8), 4, config)

    def test_too_many_channels_raises_instead_of_narrowing(self, tmp_path):
        from spectramr.pipelines.infer import _adapt_channels

        config = _norm_arm(tmp_path)
        with pytest.raises(DimensionMismatchError, match="narrowing down discards them"):
            _adapt_channels(torch.rand(1, 5, 8, 8), 3, config)

    def test_an_even_multicoil_count_raises_instead_of_rss(self, tmp_path):
        """The branch that looked like physics: 8ch -> 2ch via IFFT/RSS/FFT."""
        from spectramr.pipelines.infer import _adapt_channels

        config = _norm_arm(tmp_path)
        with pytest.raises(DimensionMismatchError, match=r"data\.coils\.processing_mode"):
            _adapt_channels(torch.rand(1, 8, 8, 8), 2, config)


class _ChannelWitness(nn.Module):
    """A model that declares ``in_channels``, so ``_get_model_channels`` reads
    it rather than silently defaulting to 1."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestPreprocessTensorMatchesTraining:
    """The end-to-end invariant, through the entry point the pipeline calls.

    These go through ``_preprocess_tensor`` -- which exists both before and
    after this change -- so they fail on the pre-fix tree for a *behavioural*
    reason, not merely because a new helper is absent.
    """

    def test_preprocess_skips_normalization_when_kspace_is_normalized(self, tmp_path):
        """58/58 cohort arms are in this state; the old code windowed 46 of them."""
        from spectramr.pipelines.infer import _preprocess_tensor

        config = _norm_arm(
            tmp_path,
            normalization_type="robust_percentile",
            enable_kspace_normalization=True,
        )
        tensor = torch.rand(1, 2, 8, 8) * 100.0
        out = _preprocess_tensor(tensor.clone(), config, torch.device("cpu"), _ChannelWitness(2))
        assert torch.equal(out, tensor), (
            "training preserves complex k-space untouched on these arms; the "
            "old inference path applied a clamp+rescale window training never "
            "applied, so every cross-verb metric compared different scalings"
        )

    def test_preprocess_matches_the_training_transform(self, tmp_path):
        """Same tensor, same declared arm, same numbers as the training chain."""
        import torchio as tio

        from spectramr.data.transforms.normalization import (
            ImageNormalizationSpec,
            ImageNormalizationTransform,
        )
        from spectramr.pipelines.infer import _preprocess_tensor

        config = _norm_arm(
            tmp_path,
            normalization_type="robust_percentile",
            enable_kspace_normalization=False,
        )
        tensor = torch.rand(1, 2, 8, 8) * 100.0

        spec = ImageNormalizationSpec.from_declared(
            "percentile",
            config.data.dataset_type,
            config.data.processing.normalization_kwargs,
        )
        subject = tio.Subject(input=tio.ScalarImage(tensor=tensor[0].unsqueeze(-1).clone()))
        expected = ImageNormalizationTransform(spec)(subject)["input"].data

        actual = _preprocess_tensor(tensor.clone(), config, torch.device("cpu"), _ChannelWitness(2))
        assert torch.allclose(actual[0].unsqueeze(-1), expected, atol=1e-6), (
            "predict must see exactly what training saw"
        )


class TestFailuresAreClassifiedByBlastRadius:
    """A run-invariant fault must abort the run, not repeat once per file.

    The loop this guards used to catch bare ``Exception`` for every input, log
    it, and continue -- so the strictness the two preprocessing contracts rely on
    (:func:`_adapt_channels`, :func:`_normalize_like_training`) never reached the
    caller. A `predict` run whose config declared an architecture the inputs
    could not satisfy logged one identical error per file, reported "Processed
    0 file(s)", and exited 0 (pitfall #16).

    These drive ``_process_all_files`` directly: nothing else in this module can
    reach the loop without a real checkpoint and a real dataset, which is why the
    behaviour went unpinned in the first place.
    """

    @staticmethod
    def _results() -> dict:
        return {"outputs": [], "num_processed": 0, "failures": []}

    @staticmethod
    def _run(monkeypatch, raiser, n_files: int, results: dict):
        import spectramr.pipelines.infer as infer_mod

        seen: list[Path] = []

        def _stub(input_file, *args, **kwargs):
            seen.append(input_file)
            raiser(input_file)

        monkeypatch.setattr(infer_mod, "_process_single_file", _stub)
        files = [Path(f"case_{i}.h5") for i in range(n_files)]
        config = SimpleNamespace(logging=LoggingConfigSchema())
        infer_mod._process_all_files(files, Path("/out"), None, None, None, config, 1, results)
        return seen

    def test_a_run_invariant_fault_aborts_on_the_first_file(self, monkeypatch) -> None:
        """The channel contract is a property of the model, not of the input."""
        results = self._results()

        def _raise(_f):
            raise DimensionMismatchError("Input has 1 channel(s) but the model")

        with pytest.raises(DimensionMismatchError):
            self._run(monkeypatch, _raise, 5, results)

    def test_a_config_fault_aborts_too_because_every_file_shares_the_config(
        self, monkeypatch
    ) -> None:
        """The second contract the loop's docstring names.

        ``_normalize_like_training`` resolves the image-normalization spec out
        of ``data.processing``, so an unrecognised ``normalization_type`` fails
        identically on every input. The resolver raises a bare ``ValueError``,
        which is *not* a ``SpectraMRError``; the pipeline retypes it to
        ``ConfigurationError`` precisely so it is classified here rather than
        swallowed once per file by the per-file handler.
        """
        results = self._results()

        def _raise(_f):
            raise ConfigurationError("[NORMALIZATION] Unknown normalization_type")

        with pytest.raises(ConfigurationError):
            self._run(monkeypatch, _raise, 5, results)
        assert results["failures"] == [], (
            "a run-invariant fault must abort, not be filed as one file's failure"
        )

    def test_it_does_not_retry_the_same_verdict_on_every_remaining_file(self, monkeypatch) -> None:
        import spectramr.pipelines.infer as infer_mod

        seen: list[Path] = []

        def _stub(input_file, *args, **kwargs):
            seen.append(input_file)
            raise DimensionMismatchError("model contract")

        monkeypatch.setattr(infer_mod, "_process_single_file", _stub)
        with pytest.raises(DimensionMismatchError):
            infer_mod._process_all_files(
                [Path(f"c_{i}.h5") for i in range(5)],
                Path("/out"),
                None,
                None,
                None,
                SimpleNamespace(logging=LoggingConfigSchema()),
                1,
                self._results(),
            )
        assert len(seen) == 1, (
            f"aborted after {len(seen)} files; a verdict that cannot change "
            "between inputs must be reported once"
        )

    def test_a_per_file_fault_continues_and_is_recorded(self, monkeypatch) -> None:
        """One unreadable input must not kill a batch -- but must be recorded."""
        results = self._results()

        def _raise(f):
            if f.name == "case_1.h5":
                raise DataCorruptionError("truncated h5")
            results["num_processed"] += 1

        seen = self._run(monkeypatch, _raise, 3, results)
        assert len(seen) == 3, "a per-file fault must not stop the batch"
        assert results["num_processed"] == 2
        assert len(results["failures"]) == 1
        assert results["failures"][0]["error"] == "truncated h5"

    def test_a_run_that_wrote_nothing_does_not_report_success(self, monkeypatch) -> None:
        """Every file failed: returning normally is the report-success facade."""

        def _raise(_f):
            raise DataCorruptionError("truncated h5")

        with pytest.raises(RuntimeError, match="produced no output"):
            self._run(monkeypatch, _raise, 3, self._results())

    def test_a_partially_successful_run_returns_with_the_failures_recorded(
        self, monkeypatch
    ) -> None:
        results = self._results()

        def _raise(f):
            if f.name == "case_0.h5":
                raise DataCorruptionError("truncated h5")
            results["num_processed"] += 1

        self._run(monkeypatch, _raise, 3, results)
        assert results["num_processed"] == 2
        assert [x["file"] for x in results["failures"]] == ["case_0.h5"]


class TestInferEmitsTheReportArtifactContract:
    """`report` is artifact-driven, so `infer` writing nothing is what blocked it.

    These pin the three things that make the report verb usable after a
    prediction run: the metric set matches the run it describes, every case is
    offered to the evaluator, and the artifacts exist before the figures that
    read them are drawn.
    """

    ARM = (
        "experiments/inprogress/kspace_filling/attention_shootout/experiment_11_attention_none.yaml"
    )

    def _metrics_block(self):
        import yaml

        from spectramr.config.schemas.metrics import MetricsConfigSchema

        path = Path(__file__).resolve().parents[3] / self.ARM
        if not path.exists():
            pytest.skip(f"cohort arm not present: {self.ARM}")
        raw = yaml.safe_load(path.read_text())
        return MetricsConfigSchema(**(raw.get("metrics") or {}))

    def test_declared_set_is_resolved_by_the_selector_training_uses(self):
        """The anti-divergence guard.

        If `infer` grew its own resolver, a report could name a different metric
        set than the training run it is compared against, and nothing would fail.
        """
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            MetricsMixin,
        )
        from spectramr.pipelines import infer as infer_mod

        block = self._metrics_block()
        expected = MetricsMixin._extract_metrics_from_config(MetricsMixin(), block)
        got = infer_mod._declared_metrics(SimpleNamespace(metrics=block))
        assert got == expected
        assert got, "an empty declared set would make this assertion vacuous"

    def test_the_cohort_arm_declares_only_full_reference_metrics(self):
        """Records why this arm's report shows skips rather than numbers.

        Not a defect of the evaluator: nothing on the inference path constructs
        the reference these five need. If a future change makes one of them
        reference-free, this test says so out loud instead of the behaviour
        changing silently.
        """
        from spectramr.core.metrics.registry import MetricsRegistry
        from spectramr.pipelines import infer as infer_mod

        declared = infer_mod._declared_metrics(SimpleNamespace(metrics=self._metrics_block()))
        assert all(MetricsRegistry.requires_reference(m) for m in declared), {
            m: MetricsRegistry.requires_reference(m) for m in declared
        }

    def test_every_processed_case_is_offered_to_the_evaluator(self, tmp_path):
        from spectramr.pipelines import infer as infer_mod

        seen = []

        class _Spy:
            def observe(self, *, case_id, prediction, target=None):
                seen.append((case_id, tuple(prediction.shape)))

        class _Strategy:
            def infer_single(self, batch):
                return batch

        monkey = pytest.MonkeyPatch()
        try:
            monkey.setattr(infer_mod, "_load_input", lambda f, d: torch.rand(2, 1, 8, 8))
            monkey.setattr(infer_mod, "_preprocess_tensor", lambda t, c, d, m: t)
            monkey.setattr(infer_mod, "_save_output", lambda *a, **k: None)
            infer_mod._process_single_file(
                tmp_path / "subjectA.nii.gz",
                tmp_path,
                MockModel(),
                _Strategy(),
                torch.device("cpu"),
                SimpleNamespace(model=SimpleNamespace(model_type="unet", model_kwargs={})),
                batch_size=2,
                results={"outputs": [], "num_processed": 0},
                evaluator=_Spy(),
            )
        finally:
            monkey.undo()

        # `.nii.gz` stems to `subjectA.nii`; what matters is that the case is
        # identified per input file rather than collapsed into one bucket.
        assert len(seen) == 1
        assert seen[0][0].startswith("subjectA")

    def test_artifacts_are_written_before_the_reporting_hook_draws(self):
        """The F10 ordering lesson, applied to the verb that just gained the hook.

        `train.py` emitted `run_summary.json` *after* calling the hook, so two
        figures that read it soft-skipped at training time and appeared only if
        `report` was re-run by hand. Re-introducing that here would recreate the
        exact "same run, different figures" divergence in the new entry point.
        """
        import inspect

        from spectramr.pipelines import infer as infer_mod

        src = inspect.getsource(infer_mod.run_inference_pipeline)
        write_at = src.find("evaluator.write(")
        summary_at = src.find("write_inference_run_summary(")
        hook_at = src.find("maybe_run_reporting(")
        assert write_at != -1 and summary_at != -1 and hook_at != -1, (
            "the artifact/report wiring is gone from run_inference_pipeline"
        )
        assert write_at < hook_at, "final_eval.json must exist before figures draw"
        assert summary_at < hook_at, "run_summary.json must exist before figures draw"

    def test_results_carry_both_computed_and_skipped_metrics(self):
        """The docstring promises "metrics"; a skip must not read as absence."""
        import inspect

        from spectramr.pipelines import infer as infer_mod

        src = inspect.getsource(infer_mod.run_inference_pipeline)
        assert 'results["metrics"]' in src
        assert 'results["metrics_skipped"]' in src


# ---------------------------------------------------------------------------
# physics.data_consistency.apply_at_predict: what the pipeline attaches
# ---------------------------------------------------------------------------


class TestPredictDataConsistencyInputs:
    """``_process_single_file`` hands the strategy the mask and the measurement.

    The projection itself is the strategy's (``PredictDataConsistency``); the
    pipeline's job is to attach what this path can honestly supply -- the
    input's ``mask`` HDF5 dataset and, on a k-space route, the preprocessed
    input as the measurement -- and to attach NOTHING when the knob is off.
    """

    @staticmethod
    def _config(dataset_type="kspace"):
        return SimpleNamespace(
            data=SimpleNamespace(dataset_type=dataset_type),
            model=SimpleNamespace(model_type="unet", model_kwargs={}),
        )

    @staticmethod
    def _h5(tmp_path, with_mask: bool, n_slices: int = 2):
        h5py = pytest.importorskip("h5py")
        import numpy as np

        p = tmp_path / "case.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("kspace", data=np.ones((n_slices, 2, 8, 8), dtype=np.float32))
            if with_mask:
                mask = np.zeros((n_slices, 8, 8), dtype=np.float32)
                mask[..., ::2] = 1.0
                f.create_dataset("mask", data=mask)
        return p

    def test_off_attaches_nothing_and_never_opens_the_file(self, tmp_path):
        from spectramr.pipelines import infer as infer_mod

        strategy = SimpleNamespace(predict_dc=None)
        missing = tmp_path / "never_read.h5"
        assert (
            infer_mod._predict_dc_inputs(strategy, missing, torch.zeros(2, 2, 8, 8), self._config())
            is None
        )
        assert infer_mod._dc_batch_kwargs(None, torch.zeros(1, 2, 8, 8), 0, 1) == {}

    def test_on_reads_the_mask_through_the_io_ssot(self, tmp_path):
        from spectramr.pipelines import infer as infer_mod

        p = self._h5(tmp_path, with_mask=True)
        found = infer_mod._predict_dc_inputs(
            SimpleNamespace(predict_dc=object()), p, torch.zeros(2, 2, 8, 8), self._config()
        )
        assert found is not None
        assert tuple(found["mask"].shape) == (2, 8, 8)
        assert found["measurement_is_input"] is True and found["num_samples"] == 2

    def test_an_absent_mask_is_reported_as_none_not_as_another_dataset(self, tmp_path):
        from spectramr.pipelines import infer as infer_mod

        p = self._h5(tmp_path, with_mask=False)
        found = infer_mod._predict_dc_inputs(
            SimpleNamespace(predict_dc=object()), p, torch.zeros(2, 2, 8, 8), self._config()
        )
        assert found is not None and found["mask"] is None

    def test_an_image_route_has_no_measurement_to_attach(self, tmp_path):
        from spectramr.pipelines import infer as infer_mod

        p = self._h5(tmp_path, with_mask=True)
        found = infer_mod._predict_dc_inputs(
            SimpleNamespace(predict_dc=object()), p, torch.zeros(2, 2, 8, 8), self._config("nifti")
        )
        assert found is not None and found["measurement_is_input"] is False
        kwargs = infer_mod._dc_batch_kwargs(found, torch.zeros(1, 2, 8, 8), 0, 1)
        assert kwargs["measured_kspace"] is None and kwargs["mask"] is not None

    def test_a_per_slice_mask_is_sliced_with_the_batch(self):
        from spectramr.pipelines import infer as infer_mod

        mask = torch.arange(4.0).view(4, 1, 1).expand(4, 8, 8)
        found = {"mask": mask, "measurement_is_input": True, "num_samples": 4}
        batch = torch.zeros(2, 2, 8, 8)
        kwargs = infer_mod._dc_batch_kwargs(found, batch, 2, 4)
        assert torch.equal(kwargs["mask"], mask[2:4])
        assert kwargs["measured_kspace"] is batch, "the measurement IS the model's input"

    def test_a_plane_mask_is_passed_whole(self):
        from spectramr.pipelines import infer as infer_mod

        mask = torch.ones(8, 8)
        found = {"mask": mask, "measurement_is_input": True, "num_samples": 4}
        assert infer_mod._dc_batch_kwargs(found, torch.zeros(2, 2, 8, 8), 2, 4)["mask"] is mask

    def test_process_single_file_forwards_them_per_batch_when_on(self, tmp_path, monkeypatch):
        from spectramr.pipelines import infer as infer_mod

        seen: list[dict] = []

        class _Strategy:
            predict_dc = object()

            def infer_single(self, batch, **kwargs):
                seen.append({"batch": batch, **kwargs})
                return batch

        p = self._h5(tmp_path, with_mask=True, n_slices=3)
        monkeypatch.setattr(infer_mod, "_load_input", lambda f, d: torch.rand(3, 2, 8, 8))
        monkeypatch.setattr(infer_mod, "_preprocess_tensor", lambda t, c, d, m: t)
        monkeypatch.setattr(infer_mod, "_save_output", lambda *a, **k: None)
        infer_mod._process_single_file(
            p,
            tmp_path,
            MockModel(),
            _Strategy(),
            torch.device("cpu"),
            self._config(),
            batch_size=2,
            results={"outputs": [], "num_processed": 0},
        )
        assert [tuple(s["batch"].shape) for s in seen] == [(2, 2, 8, 8), (1, 2, 8, 8)]
        assert [tuple(s["mask"].shape) for s in seen] == [(2, 8, 8), (1, 8, 8)]
        assert all(s["measured_kspace"] is s["batch"] for s in seen)

    def test_process_single_file_passes_nothing_when_off(self, tmp_path, monkeypatch):
        """The off state is byte-identical: a strategy without kwargs still works."""
        from spectramr.pipelines import infer as infer_mod

        class _NoKwargs:
            predict_dc = None

            def infer_single(self, batch):
                return batch

        monkeypatch.setattr(infer_mod, "_load_input", lambda f, d: torch.rand(2, 2, 8, 8))
        monkeypatch.setattr(infer_mod, "_preprocess_tensor", lambda t, c, d, m: t)
        monkeypatch.setattr(infer_mod, "_save_output", lambda *a, **k: None)
        results = {"outputs": [], "num_processed": 0}
        infer_mod._process_single_file(
            tmp_path / "x.npy",
            tmp_path,
            MockModel(),
            _NoKwargs(),
            torch.device("cpu"),
            self._config(),
            batch_size=2,
            results=results,
        )
        assert results["num_processed"] == 1

    def test_the_run_stamps_the_provenance_beside_config_source(self):
        """Result dict and run summary both say whether DC was projected."""
        import inspect

        from spectramr.pipelines import infer as infer_mod

        src = inspect.getsource(infer_mod.run_inference_pipeline)
        assert (
            src.count('results["data_consistency_at_predict"] = strategy.predict_dc_provenance()')
            == 2
        )
        summary_call = src[src.index("write_inference_run_summary(") :]
        assert (
            '"data_consistency_at_predict": results["data_consistency_at_predict"]' in summary_call
        )


class TestPredictAcquiredLines:
    """A line-axis arm gets its acquired support from the file's ISMRMRD limits (#2572)."""

    HEADER = (
        '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding><encodingLimits>'
        "<kspace_encoding_step_1><minimum>1</minimum><maximum>6</maximum>"
        "</kspace_encoding_step_1></encodingLimits></encoding></ismrmrdHeader>"
    )

    @classmethod
    def _h5(cls, tmp_path, header: str | None):
        h5py = pytest.importorskip("h5py")

        p = tmp_path / "scan.h5"
        with h5py.File(p, "w") as f:
            f["kspace"] = [0.0]
            if header is not None:
                f["ismrmrd_header"] = header
        return p

    def test_an_arm_without_a_line_axis_is_untouched_and_never_opens_the_file(self, tmp_path):
        from spectramr.pipelines import infer as infer_mod

        for strategy in (SimpleNamespace(), SimpleNamespace(acquisition_line_dim=None)):
            assert (
                infer_mod._predict_acquired_lines(
                    strategy, tmp_path / "never_read.h5", torch.zeros(2, 2, 8, 8)
                )
                is None
            )

    @pytest.mark.parametrize(("dim", "rows"), [(-2, 8), (-1, 8)])
    def test_the_support_is_the_header_range_for_every_row(self, tmp_path, dim, rows):
        from spectramr.pipelines import infer as infer_mod

        p = self._h5(tmp_path, self.HEADER)
        lines = infer_mod._predict_acquired_lines(
            SimpleNamespace(acquisition_line_dim=dim), p, torch.zeros(3, 2, 8, 8)
        )
        assert tuple(lines.shape) == (3, rows)
        assert lines[0].nonzero().flatten().tolist() == [1, 2, 3, 4, 5, 6]

    def test_an_m4raw_count_header_resolves_through_the_data_lines(self, tmp_path):
        """maximum is the COUNT (4); the file's data occupies columns 2..5, which is the support."""
        from spectramr.pipelines import infer as infer_mod

        header = self.HEADER.replace("<minimum>1</minimum>", "<minimum>0</minimum>").replace(
            "<maximum>6</maximum>", "<maximum>4</maximum>"
        )
        p = self._h5(tmp_path, header)
        tensor = torch.zeros(3, 2, 8, 8)
        tensor[..., 2:6] = 1.0
        lines = infer_mod._predict_acquired_lines(
            SimpleNamespace(acquisition_line_dim=-1), p, tensor
        )
        assert tuple(lines.shape) == (3, 8)
        assert lines[0].nonzero().flatten().tolist() == [2, 3, 4, 5]

    def test_missing_limits_raise_naming_file_and_key(self, tmp_path):
        from spectramr.data.ismrmrd_limits import EncodingLimitsUnavailableError
        from spectramr.pipelines import infer as infer_mod

        p = self._h5(tmp_path, None)
        with pytest.raises(EncodingLimitsUnavailableError, match=r"scan\.h5.*ismrmrd_header"):
            infer_mod._predict_acquired_lines(
                SimpleNamespace(acquisition_line_dim=-2), p, torch.zeros(1, 2, 8, 8)
            )

    def test_process_single_file_hands_each_batch_its_slice_of_the_support(
        self, tmp_path, monkeypatch
    ):
        from spectramr.pipelines import infer as infer_mod

        seen: list[dict] = []

        class _Strategy:
            predict_dc = None
            acquisition_line_dim = -2

            def infer_single(self, batch, **kwargs):
                seen.append({"batch": batch, **kwargs})
                return batch

        p = self._h5(tmp_path, self.HEADER)
        # Data only on lines 1..6, the header's range: dense data outside it contradicts the header.
        data = torch.zeros(3, 2, 8, 8)
        data[:, :, 1:7, :] = torch.rand(3, 2, 6, 8)
        monkeypatch.setattr(infer_mod, "_load_input", lambda f, d: data)
        monkeypatch.setattr(infer_mod, "_preprocess_tensor", lambda t, c, d, m: t)
        monkeypatch.setattr(infer_mod, "_save_output", lambda *a, **k: None)
        infer_mod._process_single_file(
            p,
            tmp_path,
            MockModel(),
            _Strategy(),
            torch.device("cpu"),
            SimpleNamespace(
                data=SimpleNamespace(dataset_type="kspace"),
                model=SimpleNamespace(model_type="unet", model_kwargs={}),
            ),
            batch_size=2,
            results={"outputs": [], "num_processed": 0},
        )
        assert [tuple(s["measured_lines"].shape) for s in seen] == [(2, 8), (1, 8)]


def test_the_run_stamps_what_predict_did_to_kspace() -> None:
    """Result dict (before and after the loop) and run summary carry the k-space stamp (#2581)."""
    import inspect

    from spectramr.pipelines import infer as infer_mod

    src = inspect.getsource(infer_mod.run_inference_pipeline)
    assert src.count('results["kspace_at_predict"] = kspace.provenance()') == 2
    summary_call = src[src.index("write_inference_run_summary(") :]
    assert '"kspace_at_predict": results["kspace_at_predict"]' in summary_call
    loop_call = src[src.index("_process_all_files(") :]
    assert "kspace," in loop_call[: loop_call.index(")")]


def test_a_strategy_that_owns_normalisation_gets_the_volume_scale() -> None:
    """``_process_single_file`` hands each batch its slice of one per-volume scale."""
    from types import SimpleNamespace

    from spectramr.pipelines.infer import _volume_kspace_scale

    calls = {}

    class _Owner:
        owns_kspace_normalization = True

        def volume_kspace_scale(self, volume, *, per_slice):
            calls["per_slice"] = per_slice
            return torch.arange(len(volume), dtype=torch.float32)

    config = SimpleNamespace(data=SimpleNamespace(slice_level_records=True))
    scale = _volume_kspace_scale(_Owner(), torch.zeros(3, 2, 4, 4), config)
    assert scale.tolist() == [0.0, 1.0, 2.0] and calls["per_slice"] is True
    # Planted: a strategy the pipeline normalises for gets nothing here.
    assert _volume_kspace_scale(SimpleNamespace(), torch.zeros(3, 2, 4, 4), config) is None


def test_the_volume_scale_is_consumed_before_the_cold_reverse_path() -> None:
    """#2878's per-batch ``kspace_scale`` stops in ``infer``; ``run_inference`` refuses it.

    Driven through ``_process_single_file`` with the real cold-diffusion strategy and
    ``PredictKSpace``, as the pipeline runs them. The planted half shows the guard the
    pop protects: the reverse path raises on a kwarg it does not read.
    """
    from spectramr.infrastructure.inference.cold_diffusion_inference_strategy import (
        ColdDiffusionInferenceStrategy,
    )
    from spectramr.infrastructure.inference.predict_kspace import PredictKSpace
    from spectramr.pipelines import infer as infer_mod

    class _Generator(nn.Module):
        num_timesteps = 8
        _dc_method = "hard"

        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Linear(1, 1)
            self.calls: list[dict] = []

        def sample(self, measurement, mask=None, inference_timesteps=None, **kwargs):
            self.calls.append(kwargs)
            return measurement

    data = {"normalize_kspace": True, "kspace_percentile": 0.99, "dataset_type": "kspace"}
    strategy = ColdDiffusionInferenceStrategy(
        _Generator(), torch.device("cpu"), {"training": {"diffusion": {"timesteps": 8}}, "data": data}
    )
    assert strategy.kspace_norm.enabled
    config = SimpleNamespace(
        data=SimpleNamespace(slice_level_records=False, **data),
        validation=SimpleNamespace(scoring=SimpleNamespace(recorrupted_inference=None)),
        model=SimpleNamespace(model_type="unet", model_kwargs={}),
    )
    seen: list[dict] = []
    original = strategy.infer_single

    def _spy(batch, **kwargs):
        seen.append(kwargs)
        return original(batch, **kwargs)

    strategy.infer_single = _spy
    tensor = torch.rand(3, 2, 8, 8)
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(infer_mod, "_load_input", lambda f, d: tensor)
        mp.setattr(infer_mod, "_preprocess_tensor", lambda t, c, d, m: t)
        mp.setattr(infer_mod, "_save_output", lambda *a, **k: None)
        results = {"outputs": [], "num_processed": 0}
        infer_mod._process_single_file(
            Path("scan.npy"),
            Path("."),
            strategy.model,
            strategy,
            torch.device("cpu"),
            config,
            2,
            results,
            None,
            PredictKSpace(config, strategy),
        )
    finally:
        mp.undo()
    assert results["num_processed"] == 1
    assert [tuple(kw["kspace_scale"].shape) for kw in seen] == [(2,), (1,)]
    assert len(strategy.model.calls) == 2
    assert all("kspace_scale" not in call for call in strategy.model.calls)
    with pytest.raises(TypeError, match="kspace_scale"):
        strategy.run_inference(tensor[:1], kspace_scale=torch.ones(1))


class TestSaveOutputFollowsTheDeclaredDomain:
    """``_save_output`` inverse-transforms only a declared k-space output.

    Planted shape: an image-domain model emitting ``[1, 2, H, W]`` real/imag (the
    FDB baseline). The channel-parity rule treated it as k-space and saved
    ``|IFFT(image)|``, a cross-hair with a dark centre.
    """

    @staticmethod
    def _run(monkeypatch, tmp_path, output, output_is_kspace, tier=1):
        import spectramr.data.writers.output_writer as writer_mod
        import spectramr.infrastructure.training.utils.domain_inference as dom
        from spectramr.pipelines import infer as infer_mod

        captured = {}

        class _Writer:
            def __init__(self, config, output_dir):
                pass

            def write(self, tensor, subject_id, file_id):
                captured["tensor"] = tensor
                return []

        monkeypatch.setattr(writer_mod, "OutputWriter", _Writer)
        domain = "kspace" if output_is_kspace else "image"
        monkeypatch.setattr(
            dom, "resolve_output_domain", lambda cfg: dom.OutputDomainDecision(domain, tier)
        )
        cfg = SimpleNamespace(
            data=SimpleNamespace(resolve_mode=lambda m: None),
            model=SimpleNamespace(model_type="unet"),
        )
        infer_mod._save_output(output, tmp_path / "x.h5", tmp_path, cfg, torch.device("cpu"))
        return captured["tensor"]

    def test_image_output_is_saved_untransformed(self, monkeypatch, tmp_path) -> None:
        out = torch.randn(1, 2, 8, 8)
        saved = self._run(monkeypatch, tmp_path, out, output_is_kspace=False)
        assert torch.equal(saved, out[0])

    def test_kspace_output_is_inverse_transformed(self, monkeypatch, tmp_path) -> None:
        out = torch.randn(1, 2, 8, 8)
        saved = self._run(monkeypatch, tmp_path, out, output_is_kspace=True)
        assert saved.shape == (1, 8, 8)
        assert float(saved.min()) >= 0.0 and float(saved.max()) <= 1.0 + 1e-6

    def test_odd_channel_declared_kspace_output_raises(self, monkeypatch, tmp_path) -> None:
        with pytest.raises(ValueError, match="even count"):
            self._run(monkeypatch, tmp_path, torch.randn(1, 3, 8, 8), output_is_kspace=True)

    def test_odd_channel_guessed_kspace_output_is_saved_as_image(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """Tier 6 (loader serves k-space) guessed for an undeclared 1-channel image model."""
        out = torch.randn(1, 1, 8, 8)
        with caplog.at_level("WARNING", logger="spectramr.pipelines.infer"):
            saved = self._run(monkeypatch, tmp_path, out, output_is_kspace=True, tier=6)
        assert torch.equal(saved, out[0])
        assert "model.target_domain" in caplog.text and "'unet'" in caplog.text


class TestInferRefusesArmsWhoseInputIsBuiltInTheTrainingStep:
    """SSDU and EI feed the network ``A^H y``; ``infer`` has no builder for it (planted arms)."""

    _SS = Path("experiments/inprogress/self_supervised")

    @pytest.mark.parametrize(
        "arm",
        ["ssdu_m4raw_r4", "robust_ssdu_m4raw_r4", "ei_multicoil_m4raw_r4", "ei_unet_m4raw_r4"],
    )
    def test_the_arm_is_refused_with_a_named_reason(self, arm):
        from spectramr.config.settings import TrainingSettings
        from spectramr.pipelines.infer import _refuse_unservable_strategy

        config = TrainingSettings.from_yaml(str(self._SS / f"{arm}.yaml"))
        with pytest.raises(ConfigurationError, match="infer cannot serve"):
            _refuse_unservable_strategy(config)

    def test_an_image_domain_n2n_arm_is_not_refused(self):
        from spectramr.config.settings import TrainingSettings
        from spectramr.pipelines.infer import _refuse_unservable_strategy

        config = TrainingSettings.from_yaml(
            "experiments/inprogress/n2n/n2n_a_r2r_single_ex_m4raw.yaml"
        )
        _refuse_unservable_strategy(config)


def _contrast_config(model_type: str, num_contrasts: int, dataset_type: str, contrast_map=None):
    return SimpleNamespace(
        model=SimpleNamespace(model_type=model_type, model_kwargs={"num_contrasts": num_contrasts}),
        data=SimpleNamespace(
            dataset_type=dataset_type,
            multi_contrast=SimpleNamespace(contrast_map=contrast_map or {}),
        ),
    )


def test_predict_reads_the_m4raw_contrast_from_the_file_name() -> None:
    from spectramr.pipelines.infer import _predict_contrast

    cfg = _contrast_config("kspace_cold_diffusion", 3, "m4raw")
    out = _predict_contrast(cfg, Path("m4raw_FLAIR_rep1.h5"), None)
    assert out is not None and out.tolist() == [2]


def test_predict_refuses_a_contrast_model_it_cannot_condition() -> None:
    from spectramr.pipelines.infer import _predict_contrast

    cfg = _contrast_config("kspace_cold_diffusion", 2, "kspace", {"CORPD_FBK": 0, "CORPDFS_FBK": 1})
    with pytest.raises(ConfigurationError, match="--contrast-idx"):
        _predict_contrast(cfg, Path("file1000123.h5"), None)
    assert _predict_contrast(cfg, Path("file1000123.h5"), 1).tolist() == [1]


def test_predict_contrast_is_not_asked_of_other_model_types() -> None:
    from spectramr.pipelines.infer import _predict_contrast

    cfg = _contrast_config("some_gan", 3, "m4raw")
    assert _predict_contrast(cfg, Path("a_T1.h5"), None) is None
    with pytest.raises(ConfigurationError):
        _predict_contrast(cfg, Path("a_T1.h5"), 1)
