"""Tests for DataPipelineDirector's BuilderContext migration (cached-cascade WS-D).

``DataPipelineDirector`` is one of the ``(config)``-only builders migrated to
the canonical ``def __init__(self, ctx: BuilderContext)`` shape behind the
:func:`spectramr.infrastructure.builders.context.accepts_builder_context` shim.
These tests pin both construction paths:

* legacy ``DataPipelineDirector(config)`` (what ~all call sites still pass), and
* canonical ``DataPipelineDirector(BuilderContext(config=config))``,

and assert they produce equivalent state (same stored ``._config``).
"""

from __future__ import annotations

import textwrap
import types
import warnings

import pytest

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.context import BuilderContext
from spectramr.infrastructure.builders.directors.data_pipeline_director import (
    DataPipelineDirector,
    resolve_sfc_expose_flags,
    strided_validation_subset,
)
from tests.utils.data_config_stub import DataConfigStub


class TestResolveSfcExposeFlags:
    """The SFC/fMRI-keys wrapper trigger must exclude the mrixfields dataset, which
    emits field_strength itself (#9 — avoid stacking a spurious fMRI wrapper)."""

    def test_mrixfields_field_strength_does_not_trigger_wrapper(self) -> None:
        cfg = types.SimpleNamespace(
            dataset_type="mrixfields", expose_field_strength=True
        )
        flags = resolve_sfc_expose_flags(cfg)
        assert flags["expose_field_strength"] is False
        assert not any(flags.values())  # nothing else set -> wrapper NOT activated

    def test_non_mrixfields_field_strength_still_triggers(self) -> None:
        # An fMRI/SFC arm that genuinely needs the wrapper is unaffected.
        cfg = types.SimpleNamespace(dataset_type="nifti", expose_field_strength=True)
        flags = resolve_sfc_expose_flags(cfg)
        assert flags["expose_field_strength"] is True
        assert any(flags.values())

    def test_mrixfields_other_expose_keys_preserved(self) -> None:
        # Only field_strength is excluded for mrixfields; other expose_* keys (if a
        # future arm set them) are still honored.
        cfg = types.SimpleNamespace(
            dataset_type="mrixfields", expose_field_strength=True, expose_site_id=True
        )
        flags = resolve_sfc_expose_flags(cfg)
        assert flags["expose_field_strength"] is False
        assert flags["expose_site_id"] is True

    def test_defaults_all_false(self) -> None:
        flags = resolve_sfc_expose_flags(
            types.SimpleNamespace(dataset_type="synthetic")
        )
        assert not any(flags.values())

    def test_the_flags_are_real_on_a_real_settings_object(self, tmp_path) -> None:
        """Drive the ACTUAL schema, not a stand-in.

        The four tests above set FLAT ``expose_*`` attributes on a
        ``SimpleNamespace``, which has whatever attribute you give it. Phase 9a
        moved those keys into a ``data.expose:`` sub-block, so on a real
        ``DataConfigSchema`` every flat read returned ``False`` -- and because
        the read is by string it never raised. The wrapper silently stopped
        being constructed for the six arms that ask for it while these four
        tests stayed green. Only a resolved settings object can tell
        "the flags are off" from "the reader is looking in the wrong place".

        The fixture declares the sub-block directly. It used to declare the
        flat spelling and lean on the fold to relocate it, but those records
        are retired now (posture ``raise``), so a flat declaration no longer
        loads at all -- which is what
        ``test_the_retired_flat_spelling_is_refused`` pins.
        """
        import textwrap

        config_file = tmp_path / "sfc.yaml"
        config_file.write_text(textwrap.dedent("""
                config_version: '1.0'
                model: {model_type: standard_unet, in_channels: 1, out_channels: 1}
                training: {training_mode: reconstruction, epochs: 1}
                optimization: {optimizer_type: adam, learning_rate: 0.001}
                logging: {log_dir: /tmp/logs}
                data:
                  dataset_type: nifti
                  patch_size: [32, 32, 1]
                  batch_size: 2
                  expose:
                    site_id: true
                    glm_design_matrix: true
                """))
        settings = TrainingSettings.from_yaml(config_file)

        # They resolve on the canonical path...
        assert settings.data.expose.site_id is True
        assert settings.data.expose.glm_design_matrix is True
        # ...and the resolver must find them there, under their legacy key names
        # (which are the wrapper's kwarg contract).
        flags = resolve_sfc_expose_flags(settings.data)
        assert flags["expose_site_id"] is True
        assert flags["expose_glm_design_matrix"] is True
        assert any(flags.values())  # <- what actually constructs the wrapper

    def test_the_retired_flat_spelling_is_refused(self, tmp_path) -> None:
        """The other half of the same fact: the flat key is gone, not folded.

        Without this, the test above would pass equally well if the flat
        spelling still loaded -- it only asserts the canonical one works.

        ``glm_design_matrix`` specifically: retirement is per RENAMES record,
        not per block. Of the eight ``expose_*`` records only this one is
        posture ``raise``; the other seven still fold because the corpus still
        declares them. Picking any of those seven would make this test assert
        the opposite of what it says.
        """
        import textwrap

        config_file = tmp_path / "flat.yaml"
        config_file.write_text(textwrap.dedent("""
                config_version: '1.0'
                model: {model_type: standard_unet, in_channels: 1, out_channels: 1}
                training: {training_mode: reconstruction, epochs: 1}
                optimization: {optimizer_type: adam, learning_rate: 0.001}
                logging: {log_dir: /tmp/logs}
                data:
                  dataset_type: nifti
                  patch_size: [32, 32, 1]
                  batch_size: 2
                  expose_glm_design_matrix: true
                """))
        # pydantic's ValidationError subclasses ValueError; matching on the
        # base avoids an import this module does not otherwise need.
        with pytest.raises(ValueError, match="expose_glm_design_matrix"):
            TrainingSettings.from_yaml(config_file)


@pytest.fixture
def settings(tmp_path) -> TrainingSettings:
    """Minimal real ``TrainingSettings`` (synthetic data, no manifest needed)."""
    config_yaml = textwrap.dedent("""
        config_version: '1.0'
        model:
          model_type: standard_unet
          in_channels: 1
          out_channels: 1

        training:
          training_mode: reconstruction
          epochs: 1

        optimization:
          optimizer_type: adam
          learning_rate: 0.001

        logging:
          log_dir: /tmp/logs
          tensorboard_enabled: false

        data:
          dataset_type: synthetic
          patch_size: [32, 32, 1]
          batch_size: 2
        """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_yaml)
    return TrainingSettings.from_yaml(str(config_file))


def test_init_accepts_legacy_config(settings: TrainingSettings) -> None:
    """Legacy ``DataPipelineDirector(config)`` keeps working via the shim."""
    director = DataPipelineDirector(settings)
    assert director._config is settings


def test_init_accepts_builder_context(settings: TrainingSettings) -> None:
    """Canonical ``DataPipelineDirector(BuilderContext(config=...))`` works."""
    director = DataPipelineDirector(BuilderContext(config=settings))
    assert director._config is settings


def test_both_forms_produce_equivalent_state(settings: TrainingSettings) -> None:
    """Both construction shapes store the identical config SSOT object."""
    legacy = DataPipelineDirector(settings)
    canonical = DataPipelineDirector(BuilderContext(config=settings))
    assert legacy._config is canonical._config is settings


def test_legacy_path_is_silent(settings: TrainingSettings) -> None:
    """Migration shim must not emit a DeprecationWarning on the legacy form.

    The repo promotes ``spectramr.*`` ``DeprecationWarning`` to a test error;
    the default (``warn=False``) shim must stay silent so existing callers
    do not break.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DataPipelineDirector(settings)  # must not raise


def test_init_marked_as_accepting_builder_context() -> None:
    """The decorator stamps the marker the fitness check reads."""
    assert (
        getattr(DataPipelineDirector.__init__, "__accepts_builder_context__", False)
        is True
    )


def test_build_dataloaders_forwards_num_workers_to_train_queue(
    settings: TrainingSettings,
) -> None:
    """``num_workers`` must reach the *training* queue, not just the val loader.

    Regression: the train path read ``data.num_workers`` from the config and
    ignored the ``num_workers`` argument, so an explicit override silently
    affected only the validation loader. The queue config the train loader is
    built from must carry the forwarded value.
    """
    from unittest.mock import patch

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
    from spectramr.data.builders.torchio_queue_builder import TorchIOQueueBuilder

    dummy = TensorDataset(torch.zeros(4, 1, 8, 8))
    captured: dict = {}

    def _fake_create_datasets(*_args, **_kwargs):
        return dummy, dummy

    def _fake_build_train_queue(_train_ds, queue_config):
        captured["queue_config"] = queue_config
        return None, DataLoader(dummy, batch_size=1)

    with (
        patch.object(
            DatasetInstantiator, "create_datasets", staticmethod(_fake_create_datasets)
        ),
        patch.object(
            TorchIOQueueBuilder,
            "build_train_queue",
            staticmethod(_fake_build_train_queue),
        ),
    ):
        director = DataPipelineDirector(settings)
        director.build_dataloaders(num_workers=2, pin_memory=False)

    # 2 is below the default (4) and below any share the reserve leaves, so it
    # distinguishes the forwarded value from a re-read of the config.
    assert captured["queue_config"].num_workers == 2



def test_the_digital_twin_flag_reaches_both_chains(tmp_path) -> None:
    """``physics`` is a run-level block. The director handed the transform builder a
    data-only proxy, so ``physics.digital_twin.apply_as_transform`` was never read."""
    from unittest.mock import patch

    from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
    from spectramr.data.transforms.tio_physics import DigitalTwinDegradation

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent("""
        config_version: '1.0'
        model: {model_type: standard_unet, in_channels: 1, out_channels: 1}
        training: {training_mode: reconstruction, epochs: 1}
        optimization: {optimizer_type: adam, learning_rate: 0.001}
        logging: {log_dir: /tmp/logs, tensorboard_enabled: false}
        physics:
          field_strength: 0.3
          digital_twin: {apply_as_transform: true}
        data: {dataset_type: synthetic, patch_size: [32, 32, 1], batch_size: 2}
        """)
    )
    st = TrainingSettings.from_yaml(str(config_file))
    chains: dict = {}

    def _capture(_data, _train_idx, _val_idx, train_tf, val_tf, **_kw):
        chains.update(train=train_tf, val=val_tf)
        raise StopIteration

    with (
        patch.object(DatasetInstantiator, "create_datasets", staticmethod(_capture)),
        pytest.raises(StopIteration),
    ):
        DataPipelineDirector(st).build_dataloaders(num_workers=0, pin_memory=False)

    for name, chain in chains.items():
        assert any(isinstance(t, DigitalTwinDegradation) for t in chain.transforms), name


# ── F4 (smoke 2026-06-16): oracle_bssfp must self-index, not load_fastmri_splits ─


def test_self_indexed_types_include_oracle_bssfp() -> None:
    """The skip-set names oracle_bssfp alongside the synthetic / generated types."""
    from spectramr.infrastructure.builders.directors.data_pipeline_director import (
        _self_indexed_dataset_types,
    )

    skip_set = _self_indexed_dataset_types()
    assert "oracle_bssfp" in skip_set
    # ``graph_mri`` was dropped 2026-08-04: it was canonical and listed here, but
    # no branch in DatasetInstantiator ever constructed it, so this entry
    # described skip-the-manifest-pre-split behaviour for a config that could
    # never load. See test_self_indexed_types_are_all_canonical below, which
    # makes the whole set self-checking against the dataset-type SSOT.
    assert {"preprocessed", "synthetic", "npy_slice", "pde_synthetic"} <= skip_set


def test_skip_set_is_derived_not_restated() -> None:
    """The skip-set must equal the registry's self-indexed routes, both ways.

    The hand-written frozenset this replaced held 5 of 12, so 7 types fell
    through to the fastMRI H5 pre-split. Subset in one direction is not enough:
    that is exactly the shape the stale list satisfied for two months.
    """
    from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
    from spectramr.data.datasets.registry import DATASET_REGISTRY
    from spectramr.infrastructure.builders.directors.data_pipeline_director import (
        _self_indexed_dataset_types,
    )

    DatasetInstantiator._ensure_registered()
    assert DATASET_REGISTRY, "registry empty -- this test would assert nothing"
    from_registry = {n for n, e in DATASET_REGISTRY.items() if not e.indexed}
    assert _self_indexed_dataset_types() == from_registry


def test_skip_set_survives_importing_the_director_first() -> None:
    """Resolving the skip-set must not depend on who imported what first.

    ``dataset_instantiator`` (which populates the registry) is imported lazily
    inside ``build_dataloaders``, so a module-level comprehension would evaluate
    against an EMPTY registry -- an empty skip-set makes every type run the
    pre-split, regressing the 45 live synthetic / npy_slice / preprocessed arms
    that work today. Assert the accessor populates rather than assumes.
    """
    import subprocess
    import sys

    # A fresh interpreter that imports ONLY the director, never the instantiator.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from spectramr.infrastructure.builders.directors import "
            "data_pipeline_director as d; "
            "print(len(d._self_indexed_dataset_types()))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert int(proc.stdout.strip().splitlines()[-1]) > 5, (
        "skip-set resolved to <=5 names in a fresh interpreter -- the registry "
        "was not populated, so the derivation silently fell back to the drift"
    )


def _oracle_settings(tmp_path) -> TrainingSettings:
    config_yaml = textwrap.dedent("""
        config_version: '1.0'
        model:
          model_type: standard_unet
          in_channels: 1
          out_channels: 1
        training:
          training_mode: reconstruction
          epochs: 1
        optimization:
          optimizer_type: adam
          learning_rate: 0.001
        logging:
          log_dir: /tmp/logs
          tensorboard_enabled: false
        data:
          dataset_type: oracle_bssfp
          patch_size: [32, 32, 1]
          batch_size: 2
        """)
    config_file = tmp_path / "oracle_config.yaml"
    config_file.write_text(config_yaml)
    return TrainingSettings.from_yaml(str(config_file))


def test_oracle_bssfp_skips_fastmri_manifest_loading(tmp_path) -> None:
    """REGRESSION: oracle_bssfp must NOT be routed through ManifestLoader's
    fastMRI splitter (which raised at ``_build_on_the_fly_index``). It builds
    its own index in DatasetInstantiator, so the manifest-loading branch is
    skipped entirely."""
    from unittest.mock import patch

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
    from spectramr.data.builders.manifest_loader import ManifestLoader
    from spectramr.data.builders.torchio_queue_builder import TorchIOQueueBuilder

    settings = _oracle_settings(tmp_path)
    dummy = TensorDataset(torch.zeros(4, 1, 8, 8))

    def _fake_create_datasets(*_a, **_k):
        return dummy, dummy

    def _fake_build_train_queue(_train_ds, queue_config):
        return None, DataLoader(dummy, batch_size=1)

    def _must_not_call(*_a, **_k):
        raise AssertionError(
            "load_fastmri_splits called for oracle_bssfp — F4 skip-set regression"
        )

    with (
        patch.object(
            DatasetInstantiator, "create_datasets", staticmethod(_fake_create_datasets)
        ),
        patch.object(
            TorchIOQueueBuilder,
            "build_train_queue",
            staticmethod(_fake_build_train_queue),
        ),
        patch.object(
            ManifestLoader,
            "load_fastmri_splits",
            staticmethod(_must_not_call),
        ),
    ):
        director = DataPipelineDirector(settings)
        train_loader, _ = director.build_dataloaders(num_workers=0, pin_memory=False)

    assert train_loader is not None


def test_mrixfields_bypasses_tio_queue(tmp_path) -> None:
    # REGRESSION (cohort-wide): mrixfields emits plain-dict samples; routing them through
    # tio.Queue crashes (subj.get_images_names() on a dict). The director must bypass the
    # queue and use the leaf DataLoaderBuilder, exactly like npy_slice.
    import json
    import textwrap
    from unittest.mock import patch

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
    from spectramr.data.builders.torchio_queue_builder import TorchIOQueueBuilder

    manifest = {
        "data_root": "",
        "records": [
            {
                "path": f"s1_{f}.nii",
                "file_id": f"s1_{f}",
                "field_strength": f,
                "contrast": "T1w",
                "subject_id": f"s1_{f}",
                "pairing_group": f"s1_{f}|T1w",
            }
            for f in (0.1, 3.0, 7.0)
        ],
    }
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(manifest))
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(textwrap.dedent(f"""
        config_version: '1.0'
        model: {{model_type: standard_unet, in_channels: 1, out_channels: 1}}
        training: {{training_mode: reconstruction, epochs: 1}}
        optimization: {{optimizer_type: adam, learning_rate: 0.001}}
        logging: {{log_dir: /tmp/logs, tensorboard_enabled: false}}
        data:
          dataset_type: mrixfields
          index_path: {mpath}
          patch_size: [32, 32, 1]
          batch_size: 2
          mrixfields:
            pairing_policy: multi_source
            target_field: 7.0
        """))
    st = TrainingSettings.from_yaml(str(cfg_file))
    dummy = TensorDataset(torch.zeros(4, 1, 8, 8))
    called = {"queue": False}

    def _fake_create(*_a, **_k):
        return dummy, dummy

    def _fake_queue(*_a, **_k):
        called["queue"] = True
        return None, DataLoader(dummy, batch_size=1)

    with (
        patch.object(
            DatasetInstantiator, "create_datasets", staticmethod(_fake_create)
        ),
        patch.object(
            TorchIOQueueBuilder, "build_train_queue", staticmethod(_fake_queue)
        ),
    ):
        director = DataPipelineDirector(st)
        train_loader, val_loader = director.build_dataloaders()

    assert (
        called["queue"] is False
    )  # queue bypassed -> no get_images_names() crash on dicts
    assert train_loader is not None and val_loader is not None


def test_validation_loader_uses_no_worker_fanout(tmp_path) -> None:
    # REGRESSION (cohort-wide validation OOM, 2026-07): ``all_slices`` validation
    # walks every slice of every val volume, each a fresh ~226 MB decode. Building
    # the val loader with the *training* ``num_workers`` made each val worker decode
    # a SEPARATE volume in parallel at the first validation, on top of the resident
    # persistent train workers -> host-RAM cgroup oom_kill before any metric was
    # written (``num_validation_batches`` caps the loop, not the spawn-time
    # prefetch fan-out). The val loader must load in the main process
    # (``num_workers == 0``); training keeps its full fan-out.
    import json
    import textwrap
    from unittest.mock import patch

    import torch
    from torch.utils.data import TensorDataset

    from spectramr.data.builders.dataset_instantiator import DatasetInstantiator

    manifest = {
        "data_root": "",
        "records": [
            {
                "path": f"s1_{f}.nii",
                "file_id": f"s1_{f}",
                "field_strength": f,
                "contrast": "T1w",
                "subject_id": f"s1_{f}",
                "pairing_group": f"s1_{f}|T1w",
            }
            for f in (0.1, 3.0, 7.0)
        ],
    }
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(manifest))
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(textwrap.dedent(f"""
        config_version: '1.0'
        model: {{model_type: standard_unet, in_channels: 1, out_channels: 1}}
        training: {{training_mode: reconstruction, epochs: 1}}
        optimization: {{optimizer_type: adam, learning_rate: 0.001}}
        logging: {{log_dir: /tmp/logs, tensorboard_enabled: false}}
        data:
          dataset_type: mrixfields
          index_path: {mpath}
          patch_size: [32, 32, 1]
          batch_size: 2
          num_workers: 4
          persistent_workers: true
          prefetch_factor: 2
          mrixfields:
            pairing_policy: multi_source
            target_field: 7.0
        """))
    st = TrainingSettings.from_yaml(str(cfg_file))
    dummy = TensorDataset(torch.zeros(4, 1, 8, 8))

    def _fake_create(*_a, **_k):
        return dummy, dummy

    with patch.object(
        DatasetInstantiator, "create_datasets", staticmethod(_fake_create)
    ):
        director = DataPipelineDirector(st)
        train_loader, val_loader = director.build_dataloaders(num_workers=4)

    # Training keeps its full worker fan-out ...
    assert train_loader.num_workers == 4
    # ... but validation loads in the main process: no second worker pool, so no
    # parallel per-worker volume decode that OOM-kills the host at first validation.
    assert val_loader.num_workers == 0
    # persistent_workers / prefetch auto-disable at num_workers == 0.
    assert getattr(val_loader, "persistent_workers", False) is False


def test_full_sampler_bypasses_tio_queue_for_training(tmp_path) -> None:
    # Full-slice training: ``modes.train.sampler.type='full'`` (whole-volume, no patching)
    # must route TRAINING through the no-Queue DataLoaderBuilder — exactly like validation —
    # not the patch tio.Queue. Fixes the ULF 'patches of a slice' symptom + the queue
    # patch-filter silently dropping small (ADC) subjects below patch_size. Mirrors the
    # npy_slice/mrixfields bypass.
    import json
    import textwrap
    from unittest.mock import patch

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
    from spectramr.data.builders.manifest_loader import ManifestLoader
    from spectramr.data.builders.torchio_queue_builder import TorchIOQueueBuilder

    manifest = {
        "data_root": "",
        "records": [
            {
                "primary_path": "a.nii",
                "target_path": "b.nii",
                "file_id": "s1",
                "subject_id": "s1",
                "contrast": "T1w",
                "split_hint": "train",
                "pairing_status": "paired",
            }
        ],
    }
    mpath = tmp_path / "paired.json"
    mpath.write_text(json.dumps(manifest))
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(textwrap.dedent(f"""
        config_version: '1.0'
        model: {{model_type: standard_unet, in_channels: 1, out_channels: 1}}
        training: {{training_mode: reconstruction, epochs: 1}}
        optimization: {{optimizer_type: adam, learning_rate: 0.001}}
        logging: {{log_dir: /tmp/logs, tensorboard_enabled: false}}
        data:
          dataset_type: nifti_paired
          paired_manifest_path: {mpath}
          patch_size: [32, 32, 1]
          batch_size: 2
          modes:
            train:
              sampler: {{type: full}}
        """))
    st = TrainingSettings.from_yaml(str(cfg_file))
    dummy = TensorDataset(torch.zeros(4, 1, 8, 8))
    called = {"queue": False}

    def _fake_create(*_a, **_k):
        return dummy, dummy

    def _fake_queue(*_a, **_k):
        called["queue"] = True
        return None, DataLoader(dummy, batch_size=1)

    with (
        patch.object(
            ManifestLoader,
            "load_paired_nifti_splits",
            staticmethod(lambda *_a, **_k: ([], [])),
        ),
        patch.object(
            DatasetInstantiator, "create_datasets", staticmethod(_fake_create)
        ),
        patch.object(
            TorchIOQueueBuilder, "build_train_queue", staticmethod(_fake_queue)
        ),
    ):
        director = DataPipelineDirector(st)
        train_loader, val_loader = director.build_dataloaders()

    assert (
        called["queue"] is False
    )  # 'full' sampler -> patch queue bypassed for training
    assert train_loader is not None and val_loader is not None


class TestBuildDataloadersKnobs:
    """Loader-knob wiring (2026-07-02, pitfall #15)."""

    def test_num_workers_default_is_none_not_zero(self) -> None:
        """``build_dataloaders`` must resolve ``num_workers`` from config when
        the caller omits it — the old ``= 0`` default silently forced a
        single-process loader despite ``data.num_workers`` (default 4)."""
        import inspect

        sig = inspect.signature(DataPipelineDirector.build_dataloaders)
        assert sig.parameters["num_workers"].default is None

    def test_build_dataloaders_wires_persistent_and_prefetch(self) -> None:
        """Source pin: every leaf DataLoaderBuilder chain forwards
        ``persistent_workers`` and ``prefetch_factor`` (previously unwired on
        the val + npy_slice/mrixfields/full-sampler paths)."""
        import inspect

        src = inspect.getsource(DataPipelineDirector.build_dataloaders)
        assert src.count(".with_persistent_workers(_persistent_train)") == 2
        assert src.count(".with_persistent_workers(_persistent_val)") == 2
        assert src.count(".with_prefetch_factor(_prefetch)") == 4
        # Resolution reads each knob off the CANONICAL sub-block and coerces
        # the worker count to a concrete int (never None) before it reaches the
        # leaf builders / queue config.
        #
        # Pinned on the receiver (`data_config.loader.<knob>`), not on an exact
        # `getattr(...)` spelling: the old form pinned the legacy flat names and
        # went red the moment phase 9a folded them under `loader`, which says
        # nothing about whether the knob is still wired. The string-keyed half
        # of this hazard -- a `getattr`/`hasattr` left on the old name while the
        # value read moved -- is covered by
        # tests/unit/config/schemas/test_renames.py::
        # TestNoStringKeyedReadsOfFoldedNames.
        for knob in (
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
        ):
            assert (
                f'data_config.loader, "{knob}"' in src
                or f"data_config.loader.{knob}" in src
            ), f"{knob} is no longer read off data_config.loader"
        assert "num_workers = int(num_workers or 0)" in src


# ── WS4: build_dataloaders honors data.pin_memory (default None → read config) ─


def test_build_dataloaders_pin_memory_defaults_to_none():
    """The default must be ``None`` (→ read ``data.pin_memory``), not a hardcoded
    ``True`` that silently ignores an explicit ``pin_memory: false``."""
    import inspect

    sig = inspect.signature(DataPipelineDirector.build_dataloaders)
    assert sig.parameters["pin_memory"].default is None


# ── #171: capped validation samples a strided spread, not the first N slices ──


class TestStridedValidationSubset:
    """``strided_validation_subset`` spreads a CAPPED val subsample across the
    dataset so the tissue-seg metrics stop grading the first N adjacent
    (background) slices of one volume (#171). Deterministic; preserves the
    ``num_validation_batches`` compute budget; a no-op when validation is
    uncapped so full-val runs are byte-identical."""

    @staticmethod
    def _cfg(num_validation_batches=None, num_samples=None):
        # Phase 10a moved both caps under `validation.loader`; the production
        # reader takes `validation_cfg.loader.num_batches` / `.num_samples`, so a
        # flat stand-in models a shape the schema no longer has.
        return types.SimpleNamespace(
            loader=types.SimpleNamespace(
                num_batches=num_validation_batches, num_samples=num_samples
            )
        )

    def test_num_batches_cap_strides_across_the_set(self) -> None:
        ds = list(range(100))
        cfg = self._cfg(num_validation_batches=4)
        out = strided_validation_subset(ds, cfg, val_batch_size=2)
        # target = 4 batches * bs 2 = 8 kept samples.
        #
        # These values changed with B9 (#171 residual). They used to be
        # [0, 12, 24, 36, 48, 60, 72, 84] — a fixed stride of 100 // 8 = 12,
        # which stops at 84 and never grades indices 85..99. The endpoint-
        # inclusive form pins BOTH ends, so the same 8 samples now span the
        # whole set. The old expectation encoded the defect.
        assert len(out) == 8
        assert list(out.indices) == [0, 14, 28, 42, 57, 71, 85, 99]
        assert max(out.indices) == 99

    def test_num_samples_cap_when_batches_unset(self) -> None:
        ds = list(range(100))
        cfg = self._cfg(num_samples=5)
        out = strided_validation_subset(ds, cfg, val_batch_size=4)
        # Was [0, 20, 40, 60, 80] under the fixed stride — the last fifth of the
        # set was unreachable. See the note above.
        assert len(out) == 5
        assert list(out.indices) == [0, 25, 50, 74, 99]

    def test_uncapped_returns_dataset_unchanged(self) -> None:
        ds = list(range(100))
        cfg = self._cfg()  # neither cap set → full val
        assert strided_validation_subset(ds, cfg, val_batch_size=2) is ds

    def test_cap_larger_than_dataset_is_a_noop(self) -> None:
        ds = list(range(6))
        cfg = self._cfg(num_validation_batches=1000)
        assert strided_validation_subset(ds, cfg, val_batch_size=2) is ds

    def test_iterable_dataset_without_len_returns_unchanged(self) -> None:
        class _NoLen:
            def __getitem__(self, i):  # pragma: no cover - never indexed here
                return i

        ds = _NoLen()
        cfg = self._cfg(num_validation_batches=4)
        assert strided_validation_subset(ds, cfg, val_batch_size=2) is ds

    def test_none_validation_cfg_returns_unchanged(self) -> None:
        ds = list(range(100))
        assert strided_validation_subset(ds, None, val_batch_size=2) is ds

    def test_mrixfields_layout_spreads_across_containers_and_full_depth(self) -> None:
        """Semantic coverage on the mrixfields ``all_slices`` layout (PR #399).

        ``MRIXFieldsPairedDataset.__len__`` for ``slice_mode='all_slices'`` is
        ``len(_index_map)``, where ``_index_map`` is CONTAINER-MAJOR:
        ``[(c, s) for c in containers for s in foreground_slices(c)]`` — every
        foreground slice of container 0, then container 1, ... So capping the val
        loop by taking the first N *contiguous* batches grades the first ~N slices
        of ONE field-pair of ONE subject, and because foreground starts at the
        brain base those are bottom-of-stack slices — the "not central only /
        include all slices" concern behind the 2026-07-19 cohort cap.

        The flat-index tests above pin the stride arithmetic; this pins the
        property that MATTERS on the real layout: the strided prefix decodes to
        many DISTINCT containers AND spans the full base->top slice-depth range,
        not a single container and not a central band. Container extents VARY
        (real volumes differ in foreground depth) so a stride aligned to a fixed
        container size cannot collapse every pick onto the same slice index.
        """
        # 30 containers, deliberately unequal foreground-slice counts.
        fg_counts = [
            187, 203, 156, 241, 198, 172, 219, 165, 230, 144,
            211, 189, 178, 225, 160, 199, 183, 236, 151, 207,
            194, 168, 222, 175, 213, 158, 228, 190, 164, 216,
        ]  # fmt: skip
        index_map = [(c, s) for c, cnt in enumerate(fg_counts) for s in range(cnt)]
        cfg = self._cfg(num_validation_batches=8)  # target = 8 * bs(2) = 16 kept
        out = strided_validation_subset(index_map, cfg, val_batch_size=2)
        kept = list(out)
        assert len(kept) == 16

        # Coverage across field-pairs/subjects: not one contiguous slab.
        distinct_containers = {c for c, _ in kept}
        assert len(distinct_containers) >= 10, (
            "capped mrixfields validation must span many containers "
            f"(subjects/field-pairs), got {sorted(distinct_containers)}"
        )
        # Coverage across slice depth: reaches near the base AND the upper stack,
        # so it is neither central-only nor bottom-only.
        depth_frac = sorted(s / fg_counts[c] for c, s in kept)
        assert depth_frac[0] < 0.15, f"never samples the brain base: {depth_frac}"
        assert depth_frac[-1] > 0.60, f"never samples the upper stack: {depth_frac}"
        assert not all(
            0.40 <= f <= 0.60 for f in depth_frac
        ), f"collapsed to a central band: {depth_frac}"


from typing import ClassVar  # noqa: E402


class TestSliceSamplerSelection:
    """``_build_slice_sampler`` decides whether the expensive epoch order applies.

    Returning ``None`` is a genuine not-applicable (nothing to amortise), not a silent
    degradation — the caller then uses ordinary shuffling. The sampler itself raises
    when ITS preconditions are violated, so the two failure modes stay distinct.
    """

    def test_builds_a_sampler_for_a_slice_expanded_dataset(self) -> None:
        from spectramr.data.samplers import VolumeBlockedSliceSampler
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _build_slice_sampler,
        )

        class _DS:
            _index_map: ClassVar = [(0, 0), (0, 1), (1, 0), (1, 1)]
            _max_resident_volumes = 6

            def container_volume_paths(self):
                return [frozenset({"a", "b"}), frozenset({"b", "c"})]

        sampler = _build_slice_sampler(_DS())
        assert isinstance(sampler, VolumeBlockedSliceSampler)
        assert sorted(sampler) == [0, 1, 2, 3]

    def test_uses_the_datasets_resolved_budget_not_the_module_default(self) -> None:
        """The sampler must block for the budget the dataset's cache actually has.

        If they disagree the blocking buys nothing — it would pack blocks the cache
        cannot hold, which is the zero-hit-rate regime all over again.
        """
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _build_slice_sampler,
        )

        class _DS:
            _index_map: ClassVar = [(0, 0), (1, 0)]
            _max_resident_volumes = 11

            def container_volume_paths(self):
                return [frozenset({"a", "b"}), frozenset({"b", "c"})]

        assert _build_slice_sampler(_DS())._max_resident == 11

    def test_returns_none_when_there_is_nothing_to_amortise(self) -> None:
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _build_slice_sampler,
        )

        class _Central:
            _index_map: ClassVar[list] = []
            _max_resident_volumes = 6

            def container_volume_paths(self):
                return []

        class _Plain:
            pass

        assert _build_slice_sampler(_Central()) is None
        assert _build_slice_sampler(_Plain()) is None


def test_self_indexed_types_are_all_canonical() -> None:
    """The skip-set must not name a type that cannot exist.

    It listed ``graph_mri``, which no branch in ``DatasetInstantiator`` ever
    constructed -- so the entry described skip-the-manifest-pre-split behaviour
    for a config that could never load.
    """
    from spectramr.config.schemas.data import CANONICAL_DATASET_TYPES
    from spectramr.infrastructure.builders.directors.data_pipeline_director import (
        _self_indexed_dataset_types,
    )

    unknown = _self_indexed_dataset_types() - set(CANONICAL_DATASET_TYPES)
    assert not unknown, f"skip-set names non-canonical types: {unknown}"


def test_presplit_raises_for_a_self_indexed_type_that_is_not_skipped(tmp_path) -> None:
    """The consequence the skip prevents: an unusable dataset type.

    This is the half a membership assertion cannot show. A self-indexed type
    absent from the skip-set falls through to ``load_fastmri_splits``, which
    globs ``*.h5``; on an arm carrying no ``index_path`` it raises before the
    arm's own creator is ever called. ``cine`` sat in exactly that state -- its
    creator globs ``*4d.nii.gz`` from ``source.root`` and works, but no config
    shape could reach it.
    """
    import nibabel as nib
    import numpy as np

    from spectramr.data.builders.manifest_loader import ManifestLoader
    from spectramr.data.datasets.cine_dataset import build_cine_index
    from spectramr.infrastructure.builders.directors.data_pipeline_director import (
        _self_indexed_dataset_types,
    )

    nib.save(
        nib.Nifti1Image(np.zeros((8, 8, 3, 5), dtype="float32"), np.eye(4)),
        str(tmp_path / "patient001_4d.nii.gz"),
    )
    # The creator handles this root fine -- the pre-split is the only blocker.
    assert len(build_cine_index(str(tmp_path))) == 1

    cfg = DataConfigStub(dataset_type="cine", data_root=str(tmp_path))
    cfg.datasets = []
    with pytest.raises((ValueError, RuntimeError)):
        ManifestLoader.load_fastmri_splits(cfg)

    # ...which is precisely why the director must not route cine here.
    assert "cine" in _self_indexed_dataset_types()
# ── D22/D7: validation.loader.num_workers, and why its guard runs EARLY ────────


class TestSharesExpensiveVolumeDecode:
    """``_shares_expensive_volume_decode`` is the one predicate behind two
    opposite decisions: the train side blocks its epoch order by volume when it
    is true, and the validation side refuses worker fan-out when it is true.
    Extracted from ``_build_slice_sampler`` rather than re-derived, so the two
    can never disagree about what "shares a decode" means.
    """

    @staticmethod
    def _volume_backed():
        class _DS:
            _index_map: ClassVar = [(0, 0), (1, 0), (2, 1)]
            _max_resident_volumes = 6

            def container_volume_paths(self):
                return [frozenset({"a"}), frozenset({"b"})]

        return _DS()

    def test_true_for_slice_expanded_volume_containers(self) -> None:
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _shares_expensive_volume_decode,
        )

        assert _shares_expensive_volume_decode(self._volume_backed()) is True

    def test_false_without_the_container_mapping(self) -> None:
        """``npy_slice`` — samples are already independent 2-D files, so worker
        fan-out costs nothing and the OOM mechanism does not apply."""
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _shares_expensive_volume_decode,
        )

        class _NpySlice:
            pass

        assert _shares_expensive_volume_decode(_NpySlice()) is False

    def test_false_when_there_are_no_containers(self) -> None:
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _shares_expensive_volume_decode,
        )

        class _Central:
            _index_map: ClassVar[list] = []

            def container_volume_paths(self):
                return []

        assert _shares_expensive_volume_decode(_Central()) is False

    def test_a_torch_subset_hides_the_answer(self) -> None:
        """The reason the guard cannot run at the loader-construction site.

        ``strided_validation_subset`` returns a plain ``torch.utils.data.Subset``
        whenever validation is capped, and ``Subset`` forwards neither
        ``container_volume_paths`` nor ``_index_map``. Asking after that rebind
        answers "no shared decode" for a dataset that plainly has one — so the
        guard would go quiet on capped arms, which are a large share of the very
        cohort the oom_kill hit.
        """
        from torch.utils.data import Subset

        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _shares_expensive_volume_decode,
        )

        raw = self._volume_backed()
        assert _shares_expensive_volume_decode(raw) is True
        assert _shares_expensive_volume_decode(Subset(raw, [0, 1])) is False

    def test_a_wrapper_without_getattr_hides_the_answer(self) -> None:
        """Same hazard via the optional wrappers. Neither ``LazyEncodeWrapper``
        nor ``SFCConformalFMRIKeysWrapper`` defines ``__getattr__``."""
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _shares_expensive_volume_decode,
        )

        class _Wrapper:
            def __init__(self, inner):
                self.inner = inner

        assert _shares_expensive_volume_decode(_Wrapper(self._volume_backed())) is False

    def test_build_slice_sampler_agrees_with_the_predicate(self) -> None:
        """SSOT: the sampler returns None exactly when the predicate is False."""
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _build_slice_sampler,
            _shares_expensive_volume_decode,
        )

        class _Plain:
            pass

        for ds in (self._volume_backed(), _Plain()):
            assert (_build_slice_sampler(ds) is not None) == (
                _shares_expensive_volume_decode(ds)
            )


class TestValidationWorkerKnob:
    """``validation.loader.num_workers`` (D7). The default 0 is the 2026-07 fix
    for a cohort-wide cgroup oom_kill at the first validation; it is a knob
    rather than a constant because the mechanism needs samples that share an
    expensive decode, which ``npy_slice`` validation does not have."""

    def test_field_exists_and_defaults_to_zero(self) -> None:
        from spectramr.config.schemas.validation import ValidationLoaderConfigSchema

        field = ValidationLoaderConfigSchema.model_fields["num_workers"]
        assert field.default == 0

    def test_description_carries_the_oom_history(self) -> None:
        """A knob whose only safe value is the default must say WHY in the place
        a user reads before setting it, or the next person re-opens the bug."""
        from spectramr.config.schemas.validation import ValidationLoaderConfigSchema

        desc = ValidationLoaderConfigSchema.model_fields["num_workers"].description
        assert desc is not None
        for token in ("oom", "volume", "npy_slice"):
            assert token in desc.lower(), f"{token!r} missing from the description"

    def test_negative_is_rejected_by_the_schema(self) -> None:
        import pydantic

        from spectramr.config.schemas.validation import ValidationLoaderConfigSchema

        with pytest.raises(pydantic.ValidationError):
            ValidationLoaderConfigSchema(num_workers=-1)

    def test_guard_runs_before_val_ds_is_rebound(self) -> None:
        """Ordering pin, not a wiring pin.

        ``val_ds`` is rebound twice after the datasets are created -- by
        ``_apply_optional_wrappers`` and by ``strided_validation_subset`` -- and
        both rebinds destroy the attributes the predicate reads (pinned
        behaviourally above). The guard is only correct in the window between
        ``create_datasets`` and the first rebind, so the position IS the
        contract here.
        """
        import inspect

        src = inspect.getsource(DataPipelineDirector.build_dataloaders)
        # Anchor on STATEMENTS, not bare symbol names: the guard's own comment
        # names `strided_validation_subset` twenty lines before the call site,
        # so a bare-name `index()` resolves to the prose and mis-orders the
        # chain. (It did, on the first run of this test.)
        guard = src.index("if _val_num_workers > 0 and _shares_expensive")
        created = src.index("train_ds, val_ds = DatasetInstantiator.create_datasets")
        wrapped = src.index("train_ds, val_ds = self._apply_optional_wrappers")
        strided = src.index("val_ds = strided_validation_subset(")

        assert created < guard < wrapped < strided, (
            "the validation-worker guard must sit between create_datasets and "
            "the first rebind of val_ds; after either rebind it reads a Subset "
            "or a wrapper and silently never fires"
        )

    def test_knob_is_read_from_the_validation_block(self) -> None:
        import inspect

        src = inspect.getsource(DataPipelineDirector.build_dataloaders)
        assert '_val_cfg.loader, "num_workers"' in src
        assert "_val_num_workers = 0" not in src, "the hardcoded constant is back"


class TestValidationWorkerGuardActuallyFires:
    """The guard's ``raise`` must be EXECUTED by a test, not merely located by one.

    The ordering pin and the predicate tests above both stay green against a
    guard whose condition is inverted or written ``>= 0``: the pin only checks
    where the statement sits, and the predicate tests never enter the branch.
    Same standard applied to the CI gate in
    ``tests/unit/ci/test_check_dataloader_construction_ssot.py`` — a check that
    cannot fail is not a check.
    """

    @staticmethod
    def _volume_backed():
        class _DS:
            _index_map: ClassVar = [(0, 0), (1, 0)]
            _max_resident_volumes = 6

            def container_volume_paths(self):
                return [frozenset({"a"}), frozenset({"b"})]

        return _DS()

    def _director(self, tmp_path, monkeypatch, num_workers):
        """A director whose dataset creation yields a volume-backed val set.

        The worker count arrives through YAML rather than by poking the frozen
        settings object, so the test exercises the real
        config -> schema -> director path the knob has to survive.
        """
        from spectramr.data.builders import dataset_instantiator

        config_file = tmp_path / "guard.yaml"
        config_file.write_text(textwrap.dedent(f"""
                config_version: '1.0'
                model: {{model_type: standard_unet, in_channels: 1, out_channels: 1}}
                training: {{training_mode: reconstruction, epochs: 1}}
                optimization: {{optimizer_type: adam, learning_rate: 0.001}}
                logging: {{log_dir: /tmp/logs}}
                data:
                  dataset_type: synthetic
                  patch_size: [32, 32, 1]
                  batch_size: 2
                validation:
                  enabled: true
                  loader:
                    num_workers: {num_workers}
                """))
        settings = TrainingSettings.from_yaml(str(config_file))
        assert settings.validation.loader.num_workers == num_workers

        ds = self._volume_backed()
        monkeypatch.setattr(
            dataset_instantiator.DatasetInstantiator,
            "create_datasets",
            staticmethod(lambda *a, **k: (ds, ds)),
        )
        return DataPipelineDirector(settings)

    def test_nonzero_workers_on_a_volume_backed_val_set_raises(
        self, tmp_path, monkeypatch
    ) -> None:
        director = self._director(tmp_path, monkeypatch, num_workers=4)
        with pytest.raises(ValueError, match="whole-volume decode"):
            director.build_dataloaders()

    def test_the_message_names_the_knob_and_the_remedy(
        self, tmp_path, monkeypatch
    ) -> None:
        """An oom_kill surfaces hours later with no attribution; this error is the
        only place the run can say which knob caused it."""
        director = self._director(tmp_path, monkeypatch, num_workers=4)
        with pytest.raises(ValueError) as exc:
            director.build_dataloaders()
        message = str(exc.value)
        assert "validation.loader.num_workers=4" in message
        assert "oom_kill" in message
        assert "npy_slice" in message  # the case where nonzero IS safe

    def test_zero_workers_does_not_raise(self, tmp_path, monkeypatch) -> None:
        """The negative half: the default must sail past the SAME dataset.

        Without this, a guard hardcoded to raise unconditionally would pass the
        two tests above.
        """
        director = self._director(tmp_path, monkeypatch, num_workers=0)
        try:
            director.build_dataloaders()
        except Exception as exc:  # see the comment below
            # Any failure here is downstream of the guard (there is no real data
            # behind the stub), which is what "the guard did not fire" means for
            # this test. Only the guard's own message would be a regression.
            assert "whole-volume decode" not in str(exc), (
                "the guard fired at the DEFAULT worker count; every arm in the "
                "corpus would fail to build"
            )
class TestStridedSubsetReachesTheTail:
    """#171 residual (B9): the stride form left the tail unreachable.

    ``range(0, total, total // target)[:target]`` starts correctly, but the
    INTEGER stride rounds down, so it exhausts ``target`` samples before reaching
    the end. Deterministic, every epoch, on every arm — the val reference was
    computed on a prefix of the set while the config asked for a spread.
    """

    @staticmethod
    def _cfg(num_batches=None, num_samples=None):
        loader = types.SimpleNamespace(num_batches=num_batches, num_samples=num_samples)
        return types.SimpleNamespace(loader=loader)

    @staticmethod
    def _indices(out):
        return list(out.indices)

    def test_the_last_sample_is_reachable(self) -> None:
        """The regression, at the plan's numbers: 1000 samples capped to 300."""
        ds = list(range(1000))
        out = strided_validation_subset(ds, self._cfg(num_samples=300), 1)
        idx = self._indices(out)
        assert idx[0] == 0
        assert idx[-1] == 999, (
            "the tail is unreachable; the old stride form stopped at 897 and the "
            "final 10.2% of the validation set was never graded"
        )

    def test_the_compute_budget_is_preserved(self) -> None:
        """The point of #171 is to spread the SAME budget, not to enlarge it."""
        ds = list(range(1000))
        out = strided_validation_subset(ds, self._cfg(num_samples=300), 1)
        assert len(self._indices(out)) == 300

    def test_small_ratios_lose_the_most_without_the_fix(self) -> None:
        """total=10, target=3 lost 30% under the stride form (last kept was 6).

        The midpoint is 4 rather than 5 because ``round`` is half-to-even and the
        exact value is 4.5. Deterministic and harmless — pinned so a later switch
        to ``int(x + 0.5)`` is a visible decision rather than silent drift in
        which samples the val reference is computed on.
        """
        out = strided_validation_subset(list(range(10)), self._cfg(num_samples=3), 1)
        assert self._indices(out) == [0, 4, 9]

    def test_a_single_sample_does_not_divide_by_zero(self) -> None:
        """``target - 1`` is the denominator; target=1 must not raise."""
        out = strided_validation_subset(list(range(1000)), self._cfg(num_samples=1), 1)
        assert self._indices(out) == [0]

    def test_indices_are_unique_and_ordered(self) -> None:
        """Rounding can collide as target approaches total. A duplicate index
        would grade the same sample twice and silently reweight the average."""
        for total, target in ((1000, 999), (100, 97), (50, 49), (7, 5)):
            out = strided_validation_subset(
                list(range(total)), self._cfg(num_samples=target), 1
            )
            idx = self._indices(out)
            assert len(idx) == len(set(idx)), f"duplicate index at {total}/{target}"
            assert idx == sorted(idx), f"unordered at {total}/{target}"
            assert idx[-1] <= total - 1

    def test_realised_count_is_logged(self, caplog) -> None:
        """The count is a MAXIMUM once rounding collides, so the run must report
        what it actually graded rather than let the reader assume the request."""
        import logging

        with caplog.at_level(logging.INFO):
            strided_validation_subset(list(range(1000)), self._cfg(num_samples=300), 1)
        assert "Strided validation subset" in caplog.text
        assert "first=0" in caplog.text
        assert "last=999" in caplog.text

    def test_uncapped_validation_is_still_untouched(self) -> None:
        """Full-val runs stay byte-identical — no Subset, no log line."""
        ds = list(range(100))
        assert strided_validation_subset(ds, self._cfg(), 1) is ds
        assert strided_validation_subset(ds, None, 1) is ds


class TestMultiDomainOverrides:
    """``data.multi_domain.domains[]`` location overrides must reach the path
    the dataset builder actually READS.

    The bug this class exists to prevent: the overrides were applied with
    ``object.__setattr__(data_cfg, "data_root", ...)``, but ``data_root`` moved
    to ``data.source.root`` on 2026-07-31. On a Pydantic model that write
    silently creates a shadow attribute -- it reads back correctly, so an
    assertion on ``cfg.data_root`` PASSES, while the builder goes on reading
    ``cfg.source.root``. Every domain therefore loaded the same corpus under a
    different tag (pitfall #16). Assert the consumed path, never the override.
    """

    @staticmethod
    def _domains():
        from spectramr.config.schemas.data import DomainConfigSchema

        return (
            DomainConfigSchema(
                name="siteA",
                data_root="/data/A",
                index_path="/idx/A.json",
                dataset_type="fastmri",
            ),
            DomainConfigSchema(name="siteB", data_root="/data/B"),
        )

    def test_overrides_land_on_the_consumed_path(self) -> None:
        from spectramr.config.schemas.data import DataConfigSchema
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _apply_domain_overrides,
        )

        base = DataConfigSchema()
        a, b = self._domains()
        cfg_a = _apply_domain_overrides(base, a)
        cfg_b = _apply_domain_overrides(base, b)

        # source.root is what DatasetInstantiator reads.
        assert cfg_a.source.root == "/data/A"
        assert cfg_b.source.root == "/data/B"
        assert cfg_a.source.root != cfg_b.source.root
        assert cfg_a.source.index_path == "/idx/A.json"
        assert cfg_a.dataset_type == "fastmri"

    def test_unset_fields_inherit_and_parent_is_untouched(self) -> None:
        from spectramr.config.schemas.data import DataConfigSchema
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _apply_domain_overrides,
        )

        base = DataConfigSchema()
        _, b = self._domains()
        cfg_b = _apply_domain_overrides(base, b)

        # siteB declares no dataset_type/index_path -> inherit.
        assert cfg_b.dataset_type == base.dataset_type
        assert cfg_b.source.index_path == base.source.index_path
        # The frozen parent must not be mutated.
        assert base.source.root != "/data/B"

    def test_no_shadow_attribute_is_created(self) -> None:
        """The override must not leave a same-named attribute nothing reads."""
        from spectramr.config.schemas.data import DataConfigSchema
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _apply_domain_overrides,
        )

        a, _ = self._domains()
        cfg = _apply_domain_overrides(DataConfigSchema(), a)
        assert "data_root" not in cfg.__dict__
        assert "index_path" not in cfg.__dict__

    def test_stale_destination_raises(self, monkeypatch) -> None:
        """A destination that MOVES must fail loud, not write a shadow attr."""
        import pytest

        from spectramr.config.schemas.data import DataConfigSchema
        from spectramr.infrastructure.builders.directors import (
            data_pipeline_director as dpd,
        )

        monkeypatch.setitem(
            dpd._DOMAIN_OVERRIDE_DESTINATIONS, "data_root", ("source", "gone")
        )
        a, _ = self._domains()
        with pytest.raises(AttributeError, match="stale"):
            dpd._apply_domain_overrides(DataConfigSchema(), a)

    def test_every_destination_is_a_real_field_today(self) -> None:
        """The mapping itself must not rot: walk each destination for real."""
        from spectramr.config.schemas.data import DataConfigSchema
        from spectramr.infrastructure.builders.directors.data_pipeline_director import (
            _DOMAIN_OVERRIDE_DESTINATIONS,
        )

        owner_root = DataConfigSchema()
        for field, path in _DOMAIN_OVERRIDE_DESTINATIONS.items():
            owner = owner_root
            for hop in path[:-1]:
                assert hop in type(owner).model_fields, f"{field}: no '{hop}'"
                owner = getattr(owner, hop)
            assert (
                path[-1] in type(owner).model_fields
            ), f"{field} -> {'.'.join(path)} is not a real field"

    def test_single_domain_is_refused_by_the_schema(self) -> None:
        """One domain is the plain loader wearing a tag.

        The refusal lives in ``MultiDomainConfigSchema``, at construction --
        which is why ``build_multi_domain_dataloaders`` carries no arity guard
        of its own. Pinned here so a later relaxation of the validator does not
        silently make a one-domain "adaptation" run constructible.
        """
        import pytest

        from spectramr.config.schemas.data import (
            DomainConfigSchema,
            MultiDomainConfigSchema,
        )

        with pytest.raises(ValueError, match="at least 2"):
            MultiDomainConfigSchema(
                enabled=True,
                domains=[DomainConfigSchema(name="only", data_root="/data/A")],
            )


class TestWorkerClampAtTheDirectorChokePoint:
    """The declared worker count is a CEILING, resolved against the topology.

    Before this, ``num_workers`` had no topology term anywhere on the path, so a
    4-rank node spawned 4x the declared decoders against one fixed core budget:
    ``num_workers: 8`` on a 16-core allocation meant 4 trainers plus 32 decoder
    processes. That thrash is one of the reasons a multi-GPU run could come out
    SLOWER than the single-GPU baseline it was meant to beat.

    The clamp lives in the director rather than in ``torchio_queue_builder``
    because ``build_dataloaders`` unconditionally overwrites
    ``queue_config.num_workers`` further down -- a clamp applied downstream is
    dead code on the production path.
    """

    @staticmethod
    def _topology(*, world_size: int, local_world_size: int, cpus: float | None):
        from spectramr.core.topology import RunTopology

        return RunTopology(
            execution_mode="slurm",
            world_size=world_size,
            local_world_size=local_world_size,
            num_nodes=1,
            rank=0,
            local_rank=0,
            cpus_on_node=cpus,
        )

    @staticmethod
    def _run(settings, topology, monkeypatch, *, num_workers):
        """Build the loaders with the leaf builders stubbed, return the director
        and the queue config the train loader would have been built from."""
        from unittest.mock import patch

        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
        from spectramr.data.builders.torchio_queue_builder import TorchIOQueueBuilder
        from spectramr.infrastructure.builders.directors import (
            data_pipeline_director as dpd,
        )

        monkeypatch.setattr(dpd, "resolve_run_topology", lambda: topology)

        dummy = TensorDataset(torch.zeros(4, 1, 8, 8))
        captured: dict = {}

        def _fake_create_datasets(*_args, **_kwargs):
            return dummy, dummy

        def _fake_build_train_queue(_train_ds, queue_config):
            captured["queue_config"] = queue_config
            return None, DataLoader(dummy, batch_size=1)

        with (
            patch.object(
                DatasetInstantiator,
                "create_datasets",
                staticmethod(_fake_create_datasets),
            ),
            patch.object(
                TorchIOQueueBuilder,
                "build_train_queue",
                staticmethod(_fake_build_train_queue),
            ),
        ):
            director = DataPipelineDirector(settings)
            director.build_dataloaders(num_workers=num_workers, pin_memory=False)
        return director, captured["queue_config"]

    def test_four_ranks_on_sixteen_cores_clamp_eight_down_to_four(
        self, settings: TrainingSettings, monkeypatch
    ) -> None:
        # The exact shape of the reported slowdown: 4 GPUs, --cpus-per-task=16,
        # the arm declaring num_workers: 8. 32 decoders on 16 cores becomes 4
        # decoders per rank -- 16 total, which is what the node actually has.
        # The default reserve of 2 cores per rank for the main process leaves 2.
        topology = self._topology(world_size=4, local_world_size=4, cpus=16.0)
        _director, queue_config = self._run(
            settings, topology, monkeypatch, num_workers=8
        )
        assert queue_config.num_workers == 2

    def test_single_gpu_is_byte_identical(
        self, settings: TrainingSettings, monkeypatch
    ) -> None:
        # The whole point of a ceiling: an arm that already fits must not move,
        # or every single-GPU baseline in the shootout is invalidated by this PR.
        topology = self._topology(world_size=1, local_world_size=1, cpus=16.0)
        _director, queue_config = self._run(
            settings, topology, monkeypatch, num_workers=7
        )
        assert queue_config.num_workers == 7

    def test_the_clamp_never_raises_above_the_declared_value(
        self, settings: TrainingSettings, monkeypatch
    ) -> None:
        # A 64-core node must not turn ``num_workers: 2`` into 64. Several arms
        # carry a deliberately low count as an OOM fix; raising it would silently
        # re-open a bug that was closed by hand.
        topology = self._topology(world_size=1, local_world_size=1, cpus=64.0)
        _director, queue_config = self._run(
            settings, topology, monkeypatch, num_workers=2
        )
        assert queue_config.num_workers == 2

    def test_a_declared_zero_stays_zero(
        self, settings: TrainingSettings, monkeypatch
    ) -> None:
        # 0 is the deliberate "load in the main process" choice, not an unset
        # value. Clamping must never turn it into 1 -- and, in the other
        # direction, the floor of 1 for positive values is what makes the
        # ``num_workers=0 + persistent_workers=True`` torch error unreachable.
        topology = self._topology(world_size=4, local_world_size=4, cpus=16.0)
        _director, queue_config = self._run(
            settings, topology, monkeypatch, num_workers=0
        )
        assert queue_config.num_workers == 0

    def test_unknown_cpu_budget_declines_to_reduce(
        self, settings: TrainingSettings, monkeypatch
    ) -> None:
        # ``cpu_resources()`` is fail-open by contract, so ``usable_cores`` can
        # be None. Guessing a share from a number we do not have would be a
        # silent default (non-negotiable 3); the clamp declines and warns.
        topology = self._topology(world_size=4, local_world_size=4, cpus=None)
        _director, queue_config = self._run(
            settings, topology, monkeypatch, num_workers=8
        )
        assert queue_config.num_workers == 8

    def test_the_decision_is_recorded_for_provenance(
        self, settings: TrainingSettings, monkeypatch
    ) -> None:
        # "What was asked for" vs "what actually ran" has to survive to disk, or
        # a re-timed arm is uncomparable to its own history for a reason nobody
        # can reconstruct.
        topology = self._topology(world_size=4, local_world_size=4, cpus=16.0)
        director, _queue_config = self._run(
            settings, topology, monkeypatch, num_workers=8
        )
        train = director.worker_decisions["train"]
        assert train["declared"] == 8
        assert train["workers"] == 2
        assert train["reserved_cores"] == 2
        assert train["clamped"] is True
        # Validation defaults to 0 and must not be clamped into existence.
        assert director.worker_decisions["val"]["workers"] == 0


def _build_default_loaders(
    tmp_path, monkeypatch, *, persistent_yaml="", cpus=16.0, **build_kwargs
):
    """Build loaders; ``persistent_yaml`` is an optional extra ``data:`` line."""
    import json
    import textwrap
    from unittest.mock import patch

    import torch
    from torch.utils.data import TensorDataset

    from spectramr.core.topology import RunTopology
    from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
    from spectramr.infrastructure.builders.directors import (
        data_pipeline_director as dpd,
    )

    topology = RunTopology(
        execution_mode="slurm",
        world_size=1,
        local_world_size=1,
        num_nodes=1,
        rank=0,
        local_rank=0,
        cpus_on_node=cpus,
    )
    monkeypatch.setattr(dpd, "resolve_run_topology", lambda: topology)
    manifest = {
        "data_root": "",
        "records": [
            {
                "path": f"s1_{f}.nii",
                "file_id": f"s1_{f}",
                "field_strength": f,
                "contrast": "T1w",
                "subject_id": f"s1_{f}",
                "pairing_group": f"s1_{f}|T1w",
            }
            for f in (0.1, 3.0, 7.0)
        ],
    }
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(manifest))
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(textwrap.dedent(f"""
        config_version: '1.0'
        model: {{model_type: standard_unet, in_channels: 1, out_channels: 1}}
        training: {{training_mode: reconstruction, epochs: 1}}
        optimization: {{optimizer_type: adam, learning_rate: 0.001}}
        logging: {{log_dir: /tmp/logs, tensorboard_enabled: false}}
        data:
          dataset_type: mrixfields
          index_path: {mpath}
          patch_size: [32, 32, 1]
          batch_size: 2
          num_workers: 2
          {persistent_yaml}
          mrixfields:
            pairing_policy: multi_source
            target_field: 7.0
        """))
    st = TrainingSettings.from_yaml(str(cfg_file))
    dummy = TensorDataset(torch.zeros(4, 1, 8, 8))
    with patch.object(
        DatasetInstantiator, "create_datasets", staticmethod(lambda *a, **k: (dummy, dummy))
    ):
        return DataPipelineDirector(st).build_dataloaders(**build_kwargs)


def test_persistent_workers_default_on_for_train_off_for_serial_val(
    tmp_path, monkeypatch
) -> None:
    # Default flipped to True; the val loader loads in the main process (0 workers),
    # where torch raises on persistent_workers=True, so only train keeps it.
    train_loader, val_loader = _build_default_loaders(tmp_path, monkeypatch)
    assert train_loader.num_workers == 2 and train_loader.persistent_workers is True
    assert val_loader.num_workers == 0 and val_loader.persistent_workers is False


def test_reserved_cores_is_recorded_in_worker_decisions(tmp_path, monkeypatch) -> None:
    from spectramr.infrastructure.builders.directors import (
        data_pipeline_director as dpd,
    )

    captured = {}
    orig = dpd.DataPipelineDirector.build_dataloaders

    def spy(self, *a, **k):
        out = orig(self, *a, **k)
        captured["d"] = self.worker_decisions
        return out

    monkeypatch.setattr(dpd.DataPipelineDirector, "build_dataloaders", spy)
    _build_default_loaders(tmp_path, monkeypatch)
    assert captured["d"]["train"]["reserved_cores"] == 2
    # the value actually handed to the DataLoader, not the declared knob
    assert captured["d"]["train"]["persistent"] is True
    assert captured["d"]["val"]["persistent"] is False


def test_explicit_persistent_true_still_leaves_a_serial_val_loader_non_persistent(
    tmp_path, monkeypatch
) -> None:
    train_loader, val_loader = _build_default_loaders(
        tmp_path, monkeypatch, persistent_yaml="persistent_workers: true"
    )
    assert train_loader.persistent_workers is True
    assert val_loader.num_workers == 0 and val_loader.persistent_workers is False


def test_explicit_persistent_false_wins_over_the_new_default(
    tmp_path, monkeypatch
) -> None:
    train_loader, _ = _build_default_loaders(
        tmp_path, monkeypatch, persistent_yaml="persistent_workers: false"
    )
    assert train_loader.num_workers == 2 and train_loader.persistent_workers is False


def test_single_pass_caller_can_turn_persistence_off(tmp_path, monkeypatch) -> None:
    train_loader, _ = _build_default_loaders(
        tmp_path, monkeypatch, persistent_workers=False
    )
    assert train_loader.num_workers == 2 and train_loader.persistent_workers is False


def test_two_cpu_single_rank_still_keeps_one_train_worker(
    tmp_path, monkeypatch
) -> None:
    # share 2 - reserve 2 would be 0; the floor of 1 keeps the declared-positive
    # count from collapsing to a serial loader.
    train_loader, _ = _build_default_loaders(tmp_path, monkeypatch, cpus=2.0)
    assert train_loader.num_workers == 1 and train_loader.persistent_workers is True
