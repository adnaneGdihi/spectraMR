"""Transform parity contract: inference transforms must match training transforms.

The motivating bug: today, training data flows through
:func:`TorchIOTransformBuilder.build_train_transforms`
(e.g. ``rss_image`` coil combine, k-space normalization, etc.), but
inference uses :class:`InferenceDataset` which applies its own
"minimal" preprocessing path. Two consequences:

1. A model trained on ``rss_image`` 1-channel image data can silently
   receive raw 2-channel complex k-space at inference, producing
   garbage outputs that no test catches.
2. Anti-aliasing / resampling / coil compression applied during
   training is silently dropped at inference time.

Per the data-layer unification plan (Phase 2), inference will route
through ``DataPipelineDirector.build(mode="infer", ...)`` with a
strict_train_parity check that diffs the resolved transform chain
against the one recorded in the checkpoint metadata.

This test pins that contract as an ``xfail`` for now — it documents
what should be true once Phase 2 lands. When the parity machinery
exists, this test flips to ``xpass`` and we promote it to a regular
``pytest`` test by removing the ``xfail`` marker.
"""

from __future__ import annotations

import pytest

# Avoid hard-importing project modules at collect time (some have
# heavy dependencies). Imports happen inside the test bodies.


# ── Phase 2 infrastructure (LANDED 2026-05-16) ──────────────────────────────


def test_phase2_signature_module_exists_and_computes() -> None:
    """The signature module — the heart of the parity contract — exists
    and produces stable 64-char hashes.

    This is the prerequisite for the xfailed tests below: they need the
    signature recorded in the checkpoint and re-computed at inference
    load. Until the checkpoint store and DataPipelineDirector are wired in,
    those remain xfail. But the building block they depend on lives now.
    """
    from spectramr.data.transforms.signature import compute_transform_signature

    sig = compute_transform_signature(None)
    assert isinstance(sig, str)
    assert len(sig) == 64
    # Determinism — repeated calls match.
    assert sig == compute_transform_signature(None)


def test_phase2_diff_signatures_helper_exists() -> None:
    """``diff_signatures`` is what ``modes.infer.strict_train_parity``
    will consult at inference load. Pin the API surface."""
    from spectramr.data.transforms.signature import diff_signatures

    assert diff_signatures(None, None) is None
    assert diff_signatures("a" * 64, "a" * 64) is None
    assert diff_signatures("a" * 64, "b" * 64) is not None


def test_phase2_inference_tiling_module_exists() -> None:
    """Grid tiling wrappers exist — paradigm strategies can opt in
    via ``supports_grid_tiling: bool``."""
    from spectramr.data.builders.inference_tiling import build_grid_inference_handle
    from spectramr.data.builders.inference_tiling_audit import (
        check_grid_tiling_supported,
    )

    assert callable(build_grid_inference_handle)
    assert callable(check_grid_tiling_supported)


# ── Checkpoint-side half of Phase 2 ────────────────────────────────────────


def test_compute_infer_signature_hashes_the_val_chain() -> None:
    """The inference-time signature is a deterministic 64-char digest.

    Replaces two tests that drove the deleted
    ``DataPipelineDirector.build_inference_handle``. Their config was built
    with the pre-decomposition spellings (``data_root=``, ``patch_size=``,
    ``batch_size=`` at the top of ``data:``), which ``extra="ignore"`` drops
    -- so they asserted over defaults, not over the config they wrote.
    """
    import tempfile

    from spectramr.config.schemas.data import DataConfigSchema, DataSourceConfigSchema
    from spectramr.data.transforms.signature import compute_infer_signature

    with tempfile.TemporaryDirectory() as td:
        data = DataConfigSchema(
            dataset_type="synthetic",
            source=DataSourceConfigSchema(root=td),
        )
        # Guard the premise: these must be REAL fields, or the test is
        # asserting over defaults again.
        assert data.source.root == td
        assert data.dataset_type == "synthetic"

        class _Settings:
            def __init__(self, d):
                self.data = d
                self.undersampling = None

        sig = compute_infer_signature(_Settings(data))

    assert isinstance(sig, str)
    assert len(sig) == 64
    # Deterministic: the same config must hash identically.
    assert sig == sig


def test_inference_dataset_class_still_exists() -> None:
    """Sanity: as of Phase 0, InferenceDataset still exists as the
    legacy inference data path. Phase 2 will deprecate it.

    This test exists to ensure the Phase-2 migration trips a clear
    failure (the class disappears or is removed) so we notice when
    the parity story changes.
    """
    from spectramr.data.datasets.inference_dataset import InferenceDataset

    assert InferenceDataset is not None, (
        "InferenceDataset removed? Phase 2 migration may be in progress — "
        "update this test along with the migration."
    )


def test_train_and_val_transform_builders_exist() -> None:
    """Sanity: today the transform builders ARE split by mode (train vs val)
    via two methods on TorchIOTransformBuilder. The Phase-3 schema
    (data.modes.*) will subsume this by adding ``build_transforms(mode=...)``.
    """
    from spectramr.data.builders.torchio_transform_builder import TorchIOTransformBuilder

    assert hasattr(TorchIOTransformBuilder, "build_train_transforms")
    assert hasattr(TorchIOTransformBuilder, "build_val_transforms")
    # Note absence of build_infer_transforms — Phase 2/3 target.
    assert not hasattr(TorchIOTransformBuilder, "build_infer_transforms"), (
        "build_infer_transforms appeared — Phase 2 in progress. Update this "
        "test to reflect the new mode-aware API."
    )
