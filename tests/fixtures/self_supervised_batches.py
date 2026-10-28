"""Real-path batches and strategies for the ``self_supervised`` cohort's tests.

A fixture that agrees with the code under test is green and blind: #2324 stood
because ``test_reconstruction_aux_extraction.py`` planted ``kspace`` as an
independent random tensor, so it could not see that production ``kspace`` is the
fully sampled reference. These helpers therefore build every batch through what
a run builds it with -- the arm's own YAML, ``DatasetInstantiator``, the train or
val transform chain, the configured collate strategy and ``BatchAdapter`` -- on a
synthetic multi-coil phantom written to an h5 file, and every strategy through
its real constructor over a small model of the arm's registered type.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "experiments" / "inprogress" / "self_supervised"


class _UndersamplingProxy:
    """The director's ``ConfigProxy``: ``data`` plus the ``undersampling`` block."""

    def __init__(self, data: Any, undersampling: Any) -> None:
        self._data = data
        self.undersampling = undersampling

    def __getattr__(self, name: str) -> Any:
        return getattr(self._data, name)


def phantom_kspace(h: int, w: int, coils: int, *, seed: int = 0) -> np.ndarray:
    """``(C, H, W)`` centred k-space of an off-centre blob seen by smooth coils.

    The brightest image pixel sits away from the grid centre on purpose, so a
    scored target whose maximum lands at the centre can only be the DC sample.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    image = np.exp(-((yy - 0.35 * h) ** 2 + (xx - 0.6 * w) ** 2) / (0.01 * h * w))
    image = image + 0.4 * np.exp(-((yy - 0.62 * h) ** 2 + (xx - 0.3 * w) ** 2) / (0.02 * h * w))
    sens = np.stack(
        [
            (0.6 + 0.4 * np.cos(2 * np.pi * c / coils + yy / h)) * np.exp(1j * (c + 0.01 * xx))
            for c in range(coils)
        ]
    )
    noise = rng.standard_normal((coils, h, w)) + 1j * rng.standard_normal((coils, h, w))
    coil_images = sens * image[None] + 0.01 * noise
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(coil_images, axes=(-2, -1)), norm="ortho"), axes=(-2, -1)
    )
    return kspace.astype(np.complex64)


def load_arm(name: str) -> Any:
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings.from_yaml(COHORT / f"{name}.yaml")


def without_shift_store(config: Any) -> Any:
    """``config`` with no ``data.nex_shift_root``: a phantom group has no precomputed store, so the loader estimates."""
    data = config.data.model_copy(update={"nex_shift_root": None})
    return config.model_copy(update={"data": data})


def with_shape(config: Any, shape: tuple[int, int]) -> Any:
    """``config`` with ``data.sampling.patch_size`` set to ``(h, w, 1)``.

    The phantom is written at the patch size, and a patch smaller than the data
    would be padded or cropped back to the YAML's square shape, which hides a
    swapped H/W.
    """
    sampling = config.data.sampling.model_copy(update={"patch_size": [*shape, 1]})
    data = config.data.model_copy(update={"sampling": sampling})
    return config.model_copy(update={"data": data})


def _write_map_store(config: Any, root: Path, files: list[Path]) -> Any:
    """Write the arm's declared coil-map store for ``files`` under ``root``.

    Uses the precompute script's own producer, so the dataset's ``raise`` policy
    reads what the cluster job writes; returns ``config`` pointed at ``root``.
    """
    import importlib.util

    from spectramr.data.datasets.m4raw_coil_maps import group_map_path, group_repetition_files
    from spectramr.infrastructure.physics.scan_coil_calibration import DEFAULT_CALIBRATION

    spec = importlib.util.spec_from_file_location(
        "precompute_m4raw_coil_maps", ROOT / "scripts" / "precompute_m4raw_coil_maps.py"
    )
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    calibration = dict(DEFAULT_CALIBRATION)
    if config.data.coil_maps_acs_size is not None:
        calibration["acs_size"] = int(config.data.coil_maps_acs_size)
    recipe = config.data.coil_maps_recipe
    for key, group in group_repetition_files(files).items():
        out = str(group_map_path(root, key))
        script.compute_one(key, [str(p) for p in group], out, calibration, "cpu", "", recipe)
    data = config.data.model_copy(update={"coil_maps_root": str(root)})
    return config.model_copy(update={"data": data})


def _write_scoring_store(config: Any, root: Path, files: list[Path]) -> Any:
    """The all-repetitions store ``coil_maps_reference_root`` names, under ``root``."""
    scoring = config.data.model_copy(
        update={"coil_maps_recipe": "all_reps_mean", "coil_maps_acs_size": None}
    )
    written = _write_map_store(config.model_copy(update={"data": scoring}), root, files)
    data = config.data.model_copy(
        update={"coil_maps_reference_root": written.data.coil_maps_root}
    )
    return config.model_copy(update={"data": data})


def real_batches(
    config: Any,
    *,
    coils: int = 4,
    route: str | None = None,
    shape: tuple[int, int] | None = None,
    pe_zero_fill: int = 0,
) -> tuple[Any, Any]:
    """``(train TrainingBatch, val TrainingBatch)`` built the way a run builds them.

    ``route`` defaults to the arm's own (``m4raw`` for ``dataset_type: m4raw``).
    ``shape`` overrides the patch ``(h, w)``; ``pe_zero_fill`` zeroes that many
    trailing phase-encode columns of every repetition, M4Raw's partial FOV.
    """
    train, val, _ = real_batches_with_reps(
        config, coils=coils, route=route, shape=shape, pe_zero_fill=pe_zero_fill
    )
    return train, val


def real_batches_with_reps(
    config: Any,
    *,
    coils: int = 4,
    route: str | None = None,
    shape: tuple[int, int] | None = None,
    pe_zero_fill: int = 0,
) -> tuple[Any, Any, list[torch.Tensor]]:
    """:func:`real_batches` plus the raw ``(1, C, H, W)`` complex repetitions written."""
    from spectramr.data.batch_types import BatchAdapter
    from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
    from spectramr.data.builders.torchio_transform_builder import (
        TorchIOTransformBuilder,
        TorchIOTransformConfig,
    )
    from spectramr.data.collation.strategies import CollateStrategyFactory
    from spectramr.data.collation.strategy_selector import CollationStrategySelector

    if route is None:
        route = "m4raw" if config.data.dataset_type == "m4raw" else "kspace"
    if shape is not None:
        config = with_shape(config, shape)
    h, w = config.data.sampling.patch_size[:2]
    reps: list[torch.Tensor] = []
    with tempfile.TemporaryDirectory() as tmp:
        if route == "kspace":
            path = Path(tmp) / "s0.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("kspace", data=phantom_kspace(h, w, coils)[None])
            index = [{"primary_path": str(path), "slice_index": 0}]
        else:
            index = []
            for rep in ("01", "02", "03"):
                path = Path(tmp) / f"2022_T1{rep}.h5"
                # One slice: the patch sampler hands the loop depth-1 patches.
                slices = phantom_kspace(h, w, coils, seed=int(rep))[None]
                if pe_zero_fill:
                    slices[..., w - pe_zero_fill :] = 0
                reps.append(torch.from_numpy(slices))
                with h5py.File(path, "w") as f:
                    f.create_dataset("kspace", data=slices)
                index.append({"primary_path": str(path)})
            config = without_shift_store(config)
            if config.data.coil_maps_root is not None:
                files = [Path(r["primary_path"]) for r in index]
                config = _write_map_store(config, Path(tmp) / "maps", files)
            if config.data.coil_maps_reference_root is not None:
                # The scoring store (#2664): all-repetitions recipe, default calibration.
                files = [Path(r["primary_path"]) for r in index]
                config = _write_scoring_store(config, Path(tmp) / "scoring_maps", files)
        tcfg = TorchIOTransformConfig.from_training_config(
            _UndersamplingProxy(config.data, config.undersampling)
        )
        train_tf = TorchIOTransformBuilder.build_train_transforms(tcfg)
        val_tf = TorchIOTransformBuilder.build_val_transforms(tcfg)
        train_ds, val_ds = DatasetInstantiator.create_datasets(
            config.data,
            index,
            index,
            train_tf,
            val_tf,
            coil_processing=config.physics.coil_processing,
        )
        name, kwargs = CollationStrategySelector.select_strategy(
            config=config.data.collation,
            dataset_type=config.data.dataset_type,
            enable_slab_mode=config.data.sampling.enable_slab_mode,
            patch_size=config.data.sampling.patch_size,
        )
        collate = CollateStrategyFactory.create(name, **kwargs).collate
        train = BatchAdapter.from_dict(collate([train_ds[0], train_ds[0]]))
        val = BatchAdapter.from_dict(collate([val_ds[0], val_ds[0]]))
    return train, val, reps


def small_model_kwargs(config: Any) -> dict[str, Any]:
    """The arm's model kwargs with the width cut down for a CPU test."""
    kwargs = dict(config.model.model_kwargs)
    features = kwargs.get("features")
    if isinstance(features, int):
        kwargs["features"] = 8
        # UNet builds its widths from model_channels x channel_mult and refuses a
        # features value it does not build, so the declared and built widths shrink together.
        if config.model.model_type == "standard_unet":
            kwargs.update(model_channels=8, channel_mult=[1, 2])
    elif isinstance(features, list):
        kwargs["features"] = [8, 16]
    return kwargs


def _ensure_logging_service() -> None:
    """What ``tests/conftest.py`` does per test, for callers in module-scoped fixtures."""
    from spectramr.domain.interfaces.service_interfaces import ILoggingService
    from spectramr.infrastructure.di.di_container import init_container
    from tests.utils.logging_service import make_logging_service

    container = init_container()
    if not container.has_service(ILoggingService):
        container.register(ILoggingService, make_logging_service(tempfile.gettempdir()))


def build_strategy(config: Any) -> Any:
    """The arm's strategy over a small instance of its registered model, on CPU."""
    from spectramr.core.parameter_free import declares_parameter_free
    from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.losses.registry import LossRegistry
    from spectramr.models.registry import get_model_class

    populate_model_registry()
    _ensure_logging_service()
    cls = get_model_class(config.model.model_type)
    params = inspect.signature(cls.__init__).parameters
    open_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
    candidate = {
        "in_channels": config.model.in_channels,
        "out_channels": config.model.out_channels,
        **small_model_kwargs(config),
    }
    model = cls(**{k: v for k, v in candidate.items() if open_kwargs or k in params})
    declared = list(config.losses.kspace_losses or []) + list(config.losses.image_losses or [])
    losses = {entry.name: LossRegistry.get_loss_class(entry.name)() for entry in declared}
    env = TrainingEnvironment.from_components(
        config=config,
        models={"generator": model},
        # A declared parameter-free solver gets no optimizer, as in the builder.
        optimizers=(
            {}
            if declares_parameter_free(model)
            else {"opt_g": torch.optim.Adam(model.parameters(), lr=1e-3)}
        ),
        losses=losses,
        device="cpu",
    )
    strategy_cls = TrainingStrategyFactory().get_strategy_class(config)
    return strategy_cls(env=env, device=torch.device("cpu"))
