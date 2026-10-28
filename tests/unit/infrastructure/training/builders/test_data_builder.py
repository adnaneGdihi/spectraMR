"""Unit tests for DataBuilder."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.config.schemas.physics import PhysicsConfigSchema
from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.context import BuilderContext
from spectramr.infrastructure.training.builders.data_builder import DataBuilder
from tests.utils.data_config_stub import DataConfigStub


@pytest.fixture
def mock_settings():
    """Create mock settings with data config."""
    settings = MagicMock(spec=TrainingSettings)
    # A bare MagicMock answers EVERY attribute, so `data.sampling.patch_size`
    # returned a Mock rather than raising -- the reader then carried a Mock into
    # a shape check instead of a list. DataConfigStub builds the sub-blocks from
    # the real schema and routes each legacy kwarg to its canonical home
    # (patch_size -> sampling.patch_size, batch_size -> loader.batch_size,
    # num_workers -> loader.num_workers, kspace_percentile ->
    # processing.kspace_percentile).
    settings.data = DataConfigStub(
        dataset_type="universal",
        batch_size=4,
        num_workers=0,
        patch_size=[64, 64],
        kspace_percentile=0.99,
    )
    # Phase 11: the reader walks `settings.undersampling`, not `.acceleration`.
    settings.undersampling = MagicMock()
    settings.undersampling.base_acceleration = 1.0
    settings.undersampling.max_acceleration = 1.0
    settings.undersampling.center_fraction = 1.0
    # Real defaults: a Mock `physics` would answer `apply_as_transform` truthy.
    settings.physics = PhysicsConfigSchema()
    return settings


@pytest.fixture
def builder(mock_settings):
    """Create DataBuilder instance."""
    return DataBuilder(mock_settings)


class TestDataBuilder:
    """Test DataBuilder functionality."""

    def test_initialization(self, builder):
        """Test builder initialization."""
        assert builder._loaders == {}

    def test_init_accepts_legacy_config_arg(self, mock_settings):
        """Legacy ``DataBuilder(config)`` call site keeps working (Phase-0 shim).

        The ``@accepts_builder_context`` adapter normalizes the legacy
        positional ``(config)`` shape into a ``BuilderContext`` before the
        migrated ``__init__(self, ctx)`` body runs.
        """
        legacy = DataBuilder(mock_settings)
        assert legacy._config is mock_settings
        assert legacy._loaders == {}

    def test_init_accepts_builder_context(self, mock_settings):
        """Canonical ``DataBuilder(BuilderContext(config=...))`` construction."""
        ctx = BuilderContext(config=mock_settings)
        canonical = DataBuilder(ctx)
        assert canonical._config is mock_settings
        assert canonical._loaders == {}

    def test_legacy_and_context_construction_equivalent(self, mock_settings):
        """Both construction shapes produce equivalent builder state."""
        legacy = DataBuilder(mock_settings)
        canonical = DataBuilder(BuilderContext(config=mock_settings))

        assert legacy._config is canonical._config
        assert legacy._loaders == canonical._loaders == {}

    @patch(
        "spectramr.infrastructure.builders.directors.data_pipeline_director.DataPipelineDirector"
    )
    def test_build_train_val_loaders_success(self, mock_director_cls, builder):
        """Test successful creation of train/val loaders.

        DataBuilder routes through ``DataPipelineDirector.build_dataloaders``
        (Phase-7 SSOT). The previous ``ConsolidatedDatasetFactory`` path is
        deprecated and emits a ``DeprecationWarning``; the patch target
        must follow the active import.
        """
        mock_train = MagicMock()
        mock_val = MagicMock()
        mock_director = MagicMock()
        mock_director.build_dataloaders.return_value = (mock_train, mock_val)
        mock_director_cls.return_value = mock_director

        builder.build_train_val_loaders()

        assert builder._loaders["train"] == mock_train
        assert builder._loaders["val"] == mock_val
        assert "test" not in builder._loaders

    @patch(
        "spectramr.infrastructure.builders.directors.data_pipeline_director.DataPipelineDirector"
    )
    def test_the_director_partition_stamp_is_kept_for_provenance(self, mock_director_cls, builder):
        stamp = {"route": "config_owned", "subjects": {"train": ["01"], "val": ["02"], "test": []}}
        mock_director = MagicMock()
        mock_director.build_dataloaders.return_value = (MagicMock(), MagicMock())
        mock_director.split_provenance = stamp
        mock_director_cls.return_value = mock_director

        builder.build_train_val_loaders()

        assert builder.split_provenance == stamp
        assert builder.split_provenance is not stamp

    @patch("spectramr.infrastructure.training.builders.data_builder.torch")
    @patch(
        "spectramr.infrastructure.builders.directors.data_pipeline_director.DataPipelineDirector"
    )
    def test_explicit_pin_memory_false_is_honored_on_cuda(
        self, mock_director_cls, mock_torch, builder
    ):
        """Pitfall-#15 regression (2026-07-02): an explicit ``data.pin_memory:
        false`` must reach the director as ``pin_memory=False`` even on a CUDA
        box — the old code forced ``torch.cuda.is_available()`` unconditionally,
        making the knob a silent no-op."""
        mock_torch.cuda.is_available.return_value = True
        # Rebind the block rather than mutating it: the loader sub-block is
        # frozen like a live config, so an in-place set raises. It only
        # appeared to work while `data` was a MagicMock -- i.e. the test
        # could never have run against the real schema.
        builder._config.data.loader = builder._config.data.loader.model_copy(
            update={"pin_memory": False}
        )
        mock_director = MagicMock()
        mock_director.build_dataloaders.return_value = (MagicMock(), MagicMock())
        mock_director_cls.return_value = mock_director

        builder.build_train_val_loaders()

        _, kwargs = mock_director.build_dataloaders.call_args
        assert kwargs["pin_memory"] is False

    @patch("spectramr.infrastructure.training.builders.data_builder.torch")
    @patch(
        "spectramr.infrastructure.builders.directors.data_pipeline_director.DataPipelineDirector"
    )
    def test_pin_memory_true_still_gated_on_cuda(
        self, mock_director_cls, mock_torch, builder
    ):
        """``pin_memory: true`` (default) stays gated on real CUDA — no pinned
        host memory / warning on CPU-only runs."""
        mock_torch.cuda.is_available.return_value = False
        # Rebind the block rather than mutating it: the loader sub-block is
        # frozen like a live config, so an in-place set raises. It only
        # appeared to work while `data` was a MagicMock -- i.e. the test
        # could never have run against the real schema.
        builder._config.data.loader = builder._config.data.loader.model_copy(
            update={"pin_memory": True}
        )
        mock_director = MagicMock()
        mock_director.build_dataloaders.return_value = (MagicMock(), MagicMock())
        mock_director_cls.return_value = mock_director

        builder.build_train_val_loaders()

        _, kwargs = mock_director.build_dataloaders.call_args
        assert kwargs["pin_memory"] is False

    def test_build_train_val_loaders_missing_config(self, builder):
        """Test error when data config is missing."""
        builder._config.data = None
        with pytest.raises(ValueError, match="TrainingSettings.data is required"):
            builder.build_train_val_loaders()

    @patch("spectramr.infrastructure.training.builders.data_builder.Path")
    @patch("spectramr.data.datasets.universal_dataset.UniversalMRIDataset")
    def test_build_inference_loader_success(self, mock_ds, mock_path, builder):
        """Test successful creation of inference loader."""
        # Mock file system
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj
        mock_path_obj.glob.return_value = [Path("test.nii.gz")]

        # Mock transforms to avoid complex dependencies
        with (
            patch("spectramr.data.builders.TorchIOTransformBuilder.build_val_transforms"),
            patch("spectramr.data.collation.strategies.CollateStrategyFactory.create"),
        ):
            builder.build_inference_loader(Path("/tmp/input"))

        assert "inference" in builder._loaders
        assert isinstance(builder._loaders["inference"], torch.utils.data.DataLoader)

    @patch("spectramr.infrastructure.training.builders.data_builder.Path")
    @patch("spectramr.data.datasets.universal_dataset.UniversalMRIDataset")
    def test_inference_loader_is_non_persistent_despite_the_schema_default(
        self, mock_ds, mock_path, builder
    ):
        """A single-pass walk must not hold a worker pool (#194), even though
        ``data.loader.persistent_workers`` now defaults to True."""
        from spectramr.infrastructure.builders.leaf.data_builders import (
            DataLoaderBuilder,
        )

        assert builder._config.data.loader.persistent_workers is True
        mock_path.return_value.glob.return_value = [Path("test.nii.gz")]
        seen: list[bool] = []
        orig = DataLoaderBuilder.with_persistent_workers

        def _spy(self_, persistent=True):
            seen.append(persistent)
            return orig(self_, persistent)

        with (
            patch("spectramr.data.builders.TorchIOTransformBuilder.build_val_transforms"),
            patch("spectramr.data.collation.strategies.CollateStrategyFactory.create"),
            patch.object(DataLoaderBuilder, "with_persistent_workers", _spy),
        ):
            builder.build_inference_loader(Path("/tmp/input"))

        assert seen == [False]

    @patch("spectramr.infrastructure.training.builders.data_builder.Path")
    @patch("spectramr.infrastructure.training.builders.data_builder.ContrastAwarePairedDataset")
    def test_contrast_aware_inference_admits_targetless_records(
        self, mock_ds, mock_path, builder
    ):
        """Inference records have no target_path; a paired-only dataset raises on them."""
        from types import SimpleNamespace

        contrast = SimpleNamespace(
            name="c", normalization="percentile", percentile=99.5, out_range=None,
            clamp=False, keywords=[],
        )
        builder._config.data.dataset_type = "contrast_aware_paired"
        builder._config.data.input_contrast = contrast
        builder._config.data.target_contrast = contrast
        mock_path.return_value.glob.return_value = [Path("test.nii.gz")]
        with (
            patch("spectramr.data.builders.TorchIOTransformBuilder.build_val_transforms"),
            patch("spectramr.data.collation.strategies.CollateStrategyFactory.create"),
            patch("spectramr.infrastructure.training.builders.data_builder.ContrastConfig"),
        ):
            builder.build_inference_loader(Path("/tmp/input"))
        assert mock_ds.call_args.kwargs["allow_unpaired"] is True

    @patch("spectramr.infrastructure.training.builders.data_builder.Path")
    def test_inference_refuses_the_digital_twin_transform(self, mock_path, builder):
        """The twin sits in the val chain; on real acquisitions it would degrade them."""
        mock_path.return_value.glob.return_value = [Path("test.nii.gz")]
        builder._config.physics = PhysicsConfigSchema(digital_twin={"apply_as_transform": True})
        with pytest.raises(ValueError, match="apply_as_transform"):
            builder.build_inference_loader(Path("/tmp/input"))

    def test_build_inference_loader_no_files(self, builder):
        """Test error when no NIfTI files found."""
        with patch(
            "spectramr.infrastructure.training.builders.data_builder.Path"
        ) as mock_path:
            mock_path.return_value.glob.return_value = []

            with pytest.raises(ValueError, match="No NIfTI files found"):
                builder.build_inference_loader(Path("/empty"))

    def test_build_ablation_subsets_success(self, builder):
        """Test creating ablation subsets."""
        # Setup mock train loader
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100

        mock_loader = MagicMock()
        mock_loader.dataset = mock_dataset
        mock_loader.batch_size = 4
        mock_loader.num_workers = 0
        mock_loader.collate_fn = None

        builder._loaders["train"] = mock_loader

        fractions = [0.1, 0.5]
        builder.build_ablation_subsets(fractions, seed=42)

        assert "ablation_10pct" in builder._loaders
        assert "ablation_50pct" in builder._loaders

        # Check subset sizes
        loader_10 = builder._loaders["ablation_10pct"]
        assert len(loader_10.dataset) == 10

    def test_build_ablation_subsets_no_train_loader(self, builder):
        """Test error when creating subsets without train loader."""
        # Mock build_train_val_loaders to do nothing or fail
        with patch.object(builder, "build_train_val_loaders"):
            with pytest.raises(ValueError, match="Cannot create ablation subsets"):
                builder.build_ablation_subsets([0.1])

    def test_build_ablation_subsets_invalid_fraction(self, builder, caplog):
        """Test warning on invalid fractions."""
        # Setup mock train loader
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_loader = MagicMock()
        mock_loader.dataset = mock_dataset
        builder._loaders["train"] = mock_loader

        builder.build_ablation_subsets([1.5, -0.1])

        assert "Invalid fraction 1.5" in caplog.text
        assert "Invalid fraction -0.1" in caplog.text


class TestAblationSubsetsCorpusSizing:
    """Review 2026-07-01: ``build_ablation_subsets`` sized subsets against
    ``len(train_loader.dataset)``. For patch-sampled training that dataset is a
    ``tio.Queue`` whose ``len`` is the patch-buffer capacity, not the number of
    training volumes — a data-fraction ablation against it is meaningless. The
    fix fails loud for the queue path and proceeds for non-patch datasets."""

    def test_raises_on_patch_queue_dataset(self, builder):
        # Class named ``Queue`` mirrors ``torchio.Queue`` — detection is a
        # class-name check (robust against MagicMock auto-vivification).
        class Queue:
            def __len__(self):
                return 256  # patch-buffer capacity, NOT the corpus size

        loader = MagicMock()
        loader.dataset = Queue()
        builder._loaders = {"train": loader}

        with pytest.raises(NotImplementedError, match="patch-buffer"):
            builder.build_ablation_subsets([0.5])

    def test_allows_non_patch_dataset(self, builder):
        """A plain (non-queue) dataset passes the guard; empty fractions avoid
        constructing any DataLoader, so the guard itself is exercised in
        isolation."""
        loader = MagicMock()
        loader.dataset = list(range(10))  # no ``subjects_dataset`` attribute
        builder._loaders = {"train": loader}

        result = builder.build_ablation_subsets([])
        assert result is builder


class TestLoaderConstructionRoutesThroughTheLeafBuilder:
    """D22: ``DataBuilder`` no longer constructs ``DataLoader`` itself.

    Both sites now go through ``infrastructure/builders/leaf/DataLoaderBuilder``,
    which is what supplies collate selection, ``worker_init_fn`` seeding and the
    prefetch/persistent knobs each hand-rolled call had dropped a different
    subset of. ``scripts/ci/check_dataloader_construction_ssot.py`` is the
    standing gate; these pin the two behaviours a reroute could silently change.
    """

    def test_no_direct_dataloader_construction_remains(self) -> None:
        """The module-level fact, asserted with the same AST resolution the CI
        gate uses rather than a substring search — ``torch.utils.data.DataLoader``
        and a bare ``DataLoader`` are different spellings of the same defect."""
        import ast
        import inspect

        from spectramr.infrastructure.training.builders import data_builder

        tree = ast.parse(inspect.getsource(data_builder))
        constructions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "DataLoader"
                )
                or (isinstance(node.func, ast.Name) and node.func.id == "DataLoader")
            )
        ]
        assert constructions == [], (
            "DataBuilder constructs a DataLoader directly again; route it "
            "through the leaf DataLoaderBuilder (data.md SSOT)."
        )

    def test_inference_loader_pins_the_image_collate_explicitly(self) -> None:
        """The reroute must NOT hand collate selection to the leaf.

        The leaf derives a strategy from ``data.collation`` whenever no
        ``collate_fn`` is given, so dropping the explicit ``"image"`` pin would
        swap the strategy on slab-mode / non-image arms — a change to inference
        OUTPUT wearing a refactor's clothes.
        """
        import inspect

        from spectramr.infrastructure.training.builders.data_builder import DataBuilder
        src = inspect.getsource(DataBuilder.build_inference_loader)
        assert 'CollateStrategyFactory.create("image")' in src
        assert ".with_collate_fn(collate_strategy.collate)" in src

    def test_inference_loader_reads_prefetch_from_config_and_is_never_persistent(self) -> None:
        """#194: both were hardcoded (2 / True) past the schema fields that exist
        to set them.

        The negative half asserts against CODE only. The method's comment
        explains what the old hardcoded values were, so it necessarily contains
        the very literals the test forbids — a naive substring check over the raw
        source fails on the prose that documents the fix. (It did, on the first
        run of this test; the same trap caught the ordering pin in
        ``test_data_pipeline_director.py``.)
        """
        import inspect

        from spectramr.infrastructure.training.builders.data_builder import DataBuilder

        src = inspect.getsource(DataBuilder.build_inference_loader)
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )

        assert ".with_prefetch_factor(data_config.loader.prefetch_factor)" in code
        assert ".with_persistent_workers(False)" in code
        assert "prefetch_factor=2" not in code
        assert "persistent_workers=True" not in code

    def test_ablation_loader_forwards_a_concrete_collate(self) -> None:
        """torch always populates ``loader.collate_fn`` (``default_collate`` when
        the caller passed None), so a falsy value means a stand-in rather than a
        real absence. Forwarding an explicit default keeps that case identical
        instead of silently promoting it to a config-derived strategy."""
        import inspect

        from spectramr.infrastructure.training.builders.data_builder import DataBuilder

        src = inspect.getsource(DataBuilder.build_ablation_subsets)
        assert "train_loader.collate_fn" in src
        assert "default_collate" in src


class TestMultiDomainRouting:
    """``data.multi_domain.enabled`` must actually route to the balancer.

    Before this wire, ``build_multi_domain_dataloaders`` had zero callers: an
    arm declaring ``multi_domain.enabled: true`` got an ordinary single-corpus
    loader and the run reported success, so domain adaptation was advertised
    and inert (pitfall #16). The schema field's own description names the
    method as the contract.
    """

    def test_builder_consults_multi_domain(self) -> None:
        """The seam: DataBuilder must branch on the flag."""
        import inspect

        from spectramr.infrastructure.training.builders.data_builder import DataBuilder
        src = inspect.getsource(DataBuilder.build_train_val_loaders)
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        assert "multi_domain" in code, (
            "DataBuilder ignores data.multi_domain, so the knob is inert"
        )
        assert "build_multi_domain_dataloaders(" in code

    def test_enabled_flag_selects_the_balancer(self, monkeypatch) -> None:
        """enabled=true calls the balancer builder; enabled=false does not."""
        from spectramr.infrastructure.builders.directors import (
            data_pipeline_director as dpd,
        )

        calls: list[str] = []

        def _fake_multi(self, **kwargs):
            calls.append("multi")
            return object()

        def _fake_single(self, **kwargs):
            calls.append("single")
            return (object(), object())

        monkeypatch.setattr(
            dpd.DataPipelineDirector, "build_multi_domain_dataloaders", _fake_multi
        )
        monkeypatch.setattr(
            dpd.DataPipelineDirector, "build_dataloaders", _fake_single
        )

        from spectramr.config.schemas.data import (
            DataConfigSchema,
            DomainConfigSchema,
            MultiDomainConfigSchema,
        )
        from spectramr.infrastructure.builders.context import BuilderContext
        from spectramr.infrastructure.training.builders.data_builder import DataBuilder

        def _run(enabled: bool) -> list[str]:
            calls.clear()
            md = MultiDomainConfigSchema(
                enabled=enabled,
                domains=[
                    DomainConfigSchema(name="a", data_root="/data/A"),
                    DomainConfigSchema(name="b", data_root="/data/B"),
                ]
                if enabled
                else [],
            )

            class _Settings:
                def __init__(self):
                    self.data = DataConfigSchema(multi_domain=md)
                    self.validation = None

            DataBuilder(BuilderContext(config=_Settings())).build_train_val_loaders()
            return list(calls)

        assert "multi" in _run(True)
        off = _run(False)
        assert "multi" not in off and "single" in off


def test_the_test_loader_batch_size_is_owned_by_the_test_split(monkeypatch):
    """The data builder passes no batch size: the validation one would make rows batch means."""
    from types import SimpleNamespace

    from spectramr.infrastructure.builders.directors import test_split
    from spectramr.infrastructure.training.builders.data_builder import DataBuilder

    seen = {}
    monkeypatch.setattr(test_split, "declares_test_set", lambda cfg: True)
    monkeypatch.setattr(test_split, "build_test_loader", lambda cfg, **kw: seen.update(kw) or "t")
    builder = DataBuilder.__new__(DataBuilder)
    builder._config = SimpleNamespace(data=SimpleNamespace(loader=SimpleNamespace(num_workers=2)))
    builder._loaders = {}
    builder.build_test_loader()
    assert builder._loaders["test"] == "t" and "val_batch_size" not in seen
