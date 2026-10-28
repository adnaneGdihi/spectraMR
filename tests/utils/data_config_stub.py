"""A ``data:`` config stand-in that tracks the real schema's sub-blocks.

The phase-9 decomposition moves ``data:`` scalars into named sub-blocks
(``loader``, ``coils``, ``expose``, ``mrixfields``, ``split``, ...). Code under
test walks the canonical path — ``config.loader.batch_size``,
``config.split.type`` — so a ``SimpleNamespace`` that still carries those knobs
flat is a shape **no real config produces**, and every check exercised against
it is answering the wrong question.

That is not hypothetical. Phase 9a shipped with the value reads moved and the
guards left on the legacy names; the stand-ins agreed with the guards, so 31
tests stayed green over code that had silently stopped working. See
``tests/unit/config/schemas/test_renames.py::TestNoStringKeyedReadsOfFoldedNames``
for the source-side half of the same hazard.

Two rules this module exists to enforce:

1. **Blocks are instantiated from the REAL schemas, never restated.** The
   production code reads some defaults straight off ``model_fields`` precisely
   so a double cannot disagree with a live config; a stand-in that hand-writes
   them re-creates the copy-rot that guard exists to prevent. (Restating them is
   how ``max_resident_volumes`` reached ``int(None)``.)
2. **One place to extend.** A new sub-block is added here once, not at thirty
   construction sites across four files.

Usage::

    from tests.utils.data_config_stub import DataConfigStub

    cfg = DataConfigStub(dataset_type="kspace", batch_size=2)
    cfg.loader.batch_size   # 2   -- flat kwarg routed to its canonical home
    cfg.split.type          # 'auto' -- schema default, not a guess

Flat kwargs are accepted and routed through :data:`FLAT_TO_CANONICAL`, mirroring
the courtesy the schema's ``fold`` records extend to YAML: a call site may keep
its old spelling while the READ side stays canonical.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

__all__ = ["DataConfigStub", "FLAT_TO_CANONICAL", "nested_blocks"]


# Sub-block attribute name -> the schema class that defines it. Extend here when
# a phase adds a block; nothing else needs to change.
_BLOCK_SCHEMAS: dict[str, str] = {
    "loader": "DataLoaderConfigSchema",
    "coils": "CoilsConfigSchema",
    "expose": "DataExposeConfigSchema",
    "mrixfields": "MRIxFieldsDataConfigSchema",
    "split": "DataSplitConfigSchema",
    "domain": "DataDomainConfigSchema",
    "processing": "DataProcessingConfigSchema",
    "sampling": "DataSamplingConfigSchema",
    "pairing": "DataPairingConfigSchema",
    "source": "DataSourceConfigSchema",
    "staging": "DataStagingConfigSchema",
    # Cine's block. Without it, ``_create_cine_dataset``'s
    # ``getattr(config, "temporal", None)`` takes the None branch against a
    # stub and raises "requires data.temporal.enabled=true" -- a failure that
    # reads as a real defect and is only a missing stand-in.
    "temporal": "TemporalConfigSchema",
    # Always present on the real schema (enabled=False by default).
    "augmentation": "AugmentationConfigSchema",
}

# Legacy flat kwarg -> (sub-block, canonical leaf). Mirrors the `fold` records in
# ``spectramr.config.schemas.renames``; kept as data so a call site that still says
# ``batch_size=2`` lands where the reader looks.
FLAT_TO_CANONICAL: dict[str, tuple[str, str]] = {
    # loader
    "batch_size": ("loader", "batch_size"),
    "num_workers": ("loader", "num_workers"),
    "persistent_workers": ("loader", "persistent_workers"),
    "pin_memory": ("loader", "pin_memory"),
    "prefetch_factor": ("loader", "prefetch_factor"),
    "max_prefetch": ("loader", "prefetch_factor"),
    # coils
    "coil_processing_mode": ("coils", "processing_mode"),
    "num_virtual_coils": ("coils", "num_virtual_coils"),
    "svd_calibration_lines": ("coils", "svd_calibration_lines"),
    # split
    "split_strategy": ("split", "type"),
    "validation_split": ("split", "validation_fraction"),
    "holdout_site": ("split", "holdout_site"),
    "train_sites": ("split", "train_sites"),
    "holdout_subject": ("split", "holdout_subject"),
    "loso_fold": ("split", "loso_fold"),
    "max_train_subjects": ("split", "max_train_subjects"),
    "max_val_subjects": ("split", "max_val_subjects"),
    # expose
    "expose_acquisition_params": ("expose", "acquisition_params"),
    "expose_conformal_jacobian": ("expose", "conformal_jacobian"),
    "expose_cortex_flatten_grid": ("expose", "cortex_flatten_grid"),
    "expose_glm_design_matrix": ("expose", "glm_design_matrix"),
    "expose_scanner_id": ("expose", "scanner_id"),
    "expose_site_id": ("expose", "site_id"),
    "expose_field_strength": ("expose", "field_strength"),
    "expose_field_strength_target": ("expose", "field_strength_target"),
    # mrixfields
    "mrixfields_pairing_policy": ("mrixfields", "pairing_policy"),
    "mrixfields_fields": ("mrixfields", "fields"),
    "mrixfields_heldout_fields": ("mrixfields", "heldout_fields"),
    "mrixfields_target_field": ("mrixfields", "target_field"),
    "mrixfields_slice_mode": ("mrixfields", "slice_mode"),
    "mrixfields_rescale_per_image": ("mrixfields", "rescale_per_image"),
    "mrixfields_output_contrast": ("mrixfields", "output_contrast"),
    "mrixfields_max_resident_volumes": ("mrixfields", "max_resident_volumes"),
    # domain
    "output_domain": ("domain", "output"),
    "target_channels": ("domain", "target_channels"),
    "input_artifact": ("domain", "input_artifact"),
    "target_artifact": ("domain", "target_artifact"),
    "graph_type": ("domain", "graph_type"),
    "enable_graph_encoding": ("domain", "enable_graph_encoding"),
    "graph_config": ("domain", "graph_config"),
    # processing
    "normalize_kspace": ("processing", "enable_kspace_normalization"),
    "kspace_percentile": ("processing", "kspace_percentile"),
    "kspace_scale_domain": ("processing", "kspace_scale_domain"),
    "log_scaling": ("processing", "enable_log_scaling"),
    "log_scaling_center_fraction": ("processing", "log_scaling_center_fraction"),
    "normalization_type": ("processing", "normalization_type"),
    "normalization_kwargs": ("processing", "normalization_kwargs"),
    "normalize_images": ("processing", "enable_image_normalization"),
    "rescale_images": ("processing", "enable_image_rescale"),
    "rescale_range": ("processing", "rescale_range"),
    "rescale_percentiles": ("processing", "rescale_percentiles"),
    "data_range": ("processing", "data_range"),
    "transforms": ("processing", "transforms"),
    # sampling
    "patch_size": ("sampling", "patch_size"),
    "samples_per_volume": ("sampling", "samples_per_volume"),
    "queue_length": ("sampling", "queue_length"),
    "subject_prefetch": ("sampling", "subject_prefetch"),
    "enable_slab_mode": ("sampling", "enable_slab_mode"),
    "slice_2d": ("sampling", "enable_slice_2d"),
    "num_synthetic_samples": ("sampling", "num_synthetic_samples"),
    # pairing -- leaf names carried across unchanged, so flat and canonical
    # differ only by the block prefix (see the block's own docstring for why
    # `contrasts` is NOT renamed to `input_contrasts`).
    "contrasts": ("pairing", "contrasts"),
    "sessions": ("pairing", "sessions"),
    "target_contrasts": ("pairing", "target_contrasts"),
    "target_sessions": ("pairing", "target_sessions"),
    "single_contrast": ("pairing", "single_contrast"),
    "bidirectional_mode": ("pairing", "bidirectional_mode"),
    "hf_resolution": ("pairing", "hf_resolution"),
    "allow_unpaired": ("pairing", "allow_unpaired"),
    # source -- `dataset_type` is NOT here: it stays flat on `data:` (a dispatch
    # key, not a location), so a stub kwarg passes straight through.
    "data_root": ("source", "root"),
    "data_layout": ("source", "layout"),
    "index_path": ("source", "index_path"),
    "validation_index_path": ("source", "validation_index_path"),
    "paired_manifest_path": ("source", "paired_manifest_path"),
    "preprocessing_dir": ("source", "preprocessing_dir"),
}


def nested_blocks(**override: Any) -> dict[str, Any]:
    """Every phase-9 sub-block, instantiated at its real schema defaults."""
    import spectramr.config.schemas.data as _data

    blocks: dict[str, Any] = {
        name: getattr(_data, cls)() for name, cls in _BLOCK_SCHEMAS.items()
    }
    blocks.update(override)
    return blocks


class DataConfigStub(SimpleNamespace):
    """Stand-in for ``config.data`` carrying every sub-block by default.

    Explicit kwargs win over the defaults. A flat legacy kwarg is routed to its
    canonical home rather than set as a top-level attribute, so it cannot become
    a second spelling that silently disagrees with the one the reader walks.
    """

    def __init__(self, **kwargs: Any) -> None:
        blocks = nested_blocks()
        # `TorchIOTransformConfig.from_training_config` reads `config.undersampling`,
        # the phase-11 TOP-LEVEL block. Production never hands it a bare
        # `data:` object -- every call site wraps one in a `ConfigProxy` that
        # sets `self.undersampling = a_conf` (four of them, e.g.
        # data_pipeline_director.py:341). A stand-in that omits it models a
        # receiver that does not exist and fails for the wrong reason.
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema

        blocks.setdefault("undersampling", AccelerationConfigSchema())
        for flat, (block, leaf) in FLAT_TO_CANONICAL.items():
            if flat in kwargs:
                # Every sub-block is frozen=True, so copy rather than mutate.
                blocks[block] = blocks[block].model_copy(
                    update={leaf: kwargs.pop(flat)}
                )
        super().__init__(**{**blocks, **kwargs})
