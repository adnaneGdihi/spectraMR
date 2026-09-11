"""Regression tests for the federated cross-contrast validation slice.

The 2026-05-13 mosaic audit traced the experiment_11 "doubled target" PNGs
to the M4Raw cross-contrast dataloader stacking
``[source_kspace, target_kspace]`` along the channel axis. The validation
save then RSS-combined ALL 16 channels (8 source + 8 target) into one
magnitude image, mixing T1 and T2/FLAIR anatomy in a single render —
the visual "doubled brain" the user reported.

Fix scaffold:
  1. M4RawRepetitionDataset publishes ``federated_target_channel_start``
     on the subject (the channel index where the target half begins).
  2. The diffusion strategy validation logger accepts that boundary as
     ``federated_target_channel_start`` kwarg, slices both prediction
     and target tensors to ``[:, start:, ...]``, and only then runs
     ``kspace_to_image`` — so the RSS only mixes coils within one
     contrast.

The single grep-pin below reads ``m4raw_dataset.py``'s source text — the
producer half, which no test here executes. The strategy half is driven for
real instead: :func:`test_the_marker_reaches_the_logger_from_a_producer_batch`
runs ``_compute_validation_metrics`` and reads the kwarg the logger actually
received. The rest exercise
:meth:`DiffusionTrainingStrategy._resolve_federated_target_start` directly.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from spectramr.core.metrics.computer import ValidationMetricsComputer
from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig
from spectramr.data.batch_types import BatchAdapter
from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)

# The grep-pin reads a file by path, so anchor on the repo root rather than the
# process cwd — tests/unit/infrastructure/training/strategies/ -> parents[5].
REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]


# ─── Grep-pin: the dataset must publish the marker. This one stays a source
# read because nothing here constructs an M4Raw subject; the strategy side is
# pinned behaviourally below instead.


def test_m4raw_dataset_publishes_federated_split_metadata():
    """Cross-contrast M4Raw must mark the (subject) with the channel boundary.

    Without this metadata, the strategy can't recover the split and the
    visualization would RSS T1 + T2 anatomy into a single mixed render.
    """
    src = (REPO_ROOT / "src/spectramr/data/datasets/m4raw_dataset.py").read_text(
        encoding="utf-8"
    )
    assert 'subject["federated_target_channel_start"]' in src, (
        "M4RawRepetitionDataset must publish ``federated_target_channel_start`` "
        "on the cross-contrast subject so the validation save can slice the "
        "target-contrast half only."
    )


# ─── The strategy half, driven rather than grepped (#1939).
#
# ``batch_data`` reaches ``_compute_validation_metrics`` from
# ``pipelines.train.select_validation_extra_fields``, which forwards whatever
# the loader produced. Since ``train.py`` adapts every validation batch through
# ``BatchAdapter.from_dict`` that is a ``TrainingBatch``: not a mapping, and
# with every non-core key in ``.metadata``, where attribute lookup cannot see
# it. The fixtures below are therefore built through that same producer — a
# hand-assembled batch would agree by construction and prove nothing.


def _metrics_mock() -> MagicMock:
    """A ``self`` complete enough to run ``_compute_validation_metrics`` to return.

    ``output_transform="none"`` makes ``_apply_metric_transforms`` the identity,
    which isolates the seam under test (the batch read) from the transform's own
    behaviour.
    """
    mock = MagicMock()
    mock.config = SimpleNamespace(
        data=SimpleNamespace(
            processing=SimpleNamespace(
                enable_kspace_normalization=False, enable_log_scaling=False
            )
        ),
        model=SimpleNamespace(input_type="image", model_type="kspace_cold_diffusion"),
        validation=SimpleNamespace(
            scoring=SimpleNamespace(
                enable_image_metrics=True, domain="image", output_transform="none"
            )
        ),
    )
    mock._is_cold_diffusion = MagicMock(return_value=True)
    mock._apply_metric_transforms = lambda pred, target, cfg: (pred, target)
    mock._measure_prediction_scale = lambda pred, target: {
        "pred_above_target_fraction": 0.0,
        "target_abs_max": 1.0,
        "pred_target_scale_ratio": 1.0,
    }
    mock._convert_metrics_to_floats = lambda d: d
    mock._PRED_SCALE_WARN_FRACTION = DiffusionTrainingStrategy._PRED_SCALE_WARN_FRACTION
    mock._ZF_BASELINE_METRICS = DiffusionTrainingStrategy._ZF_BASELINE_METRICS
    mock.validation_metrics_computer = ValidationMetricsComputer(
        ValidationMetricsConfig(
            metrics=[MetricSpec(name="psnr")], primary_metric="psnr"
        ),
        device="cpu",
    )
    mock._zf_measurement = None
    return mock


def _forwarded_marker(batch_data):
    """Run the real method and return the marker the logger was handed."""
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    inputs = torch.rand(2, 1, 8, 8)
    mock = _metrics_mock()
    DiffusionTrainingStrategy._compute_validation_metrics(
        mock,
        pred,
        target,
        inputs,
        torch.zeros(2, dtype=torch.long),
        batch_data,
        torch.ones(2),
    )
    mock._log_validation_images_to_tensorboard.assert_called_once()
    return mock._log_validation_images_to_tensorboard.call_args.kwargs[
        "federated_target_channel_start"
    ]


def test_the_marker_reaches_the_logger_from_a_producer_batch():
    """The regression: a TrainingBatch publishing the marker must forward it.

    Both legs of the replaced ``isinstance(dict)`` / ``getattr`` pairing missed
    here — the batch is a dataclass, so the mapping leg was False, and the
    marker lives in ``.metadata``, which attribute lookup cannot reach. The
    logger received ``None`` and rendered the full 16-channel stack: the
    doubled-target PNG this module exists to prevent, silently restored.
    """
    marker = torch.tensor(8, dtype=torch.long)
    batch = BatchAdapter.from_dict(
        {
            "input": torch.rand(2, 1, 8, 8),
            "target": torch.rand(2, 1, 8, 8),
            "federated_target_channel_start": marker,
        }
    )

    # The fixture is only the hard case if BOTH legs of the old read miss it.
    # Asserted rather than assumed: a producer that grew an attribute for this
    # key would make the test below pass for the wrong reason.
    assert not isinstance(batch, dict)
    assert getattr(batch, "federated_target_channel_start", None) is None

    forwarded = _forwarded_marker(batch)
    assert forwarded is not None, (
        "the marker the batch published was read as absent — the logger will "
        "RSS both contrasts into one render"
    )
    assert int(forwarded) == 8


def test_a_dict_batch_still_forwards_the_marker():
    """``read_batch_field`` is a superset, so the dict path must be unchanged.

    Nothing in the pipeline hands this method a dict today, but the replaced
    code had a working dict leg and dropping it would be a silent narrowing.
    """
    assert int(_forwarded_marker({"federated_target_channel_start": 8})) == 8


def test_a_batch_without_the_marker_forwards_none():
    """Absent stays absent: ``None`` means "render the full stack"."""
    batch = BatchAdapter.from_dict(
        {"input": torch.rand(2, 1, 8, 8), "target": torch.rand(2, 1, 8, 8)}
    )
    assert _forwarded_marker(batch) is None
    assert _forwarded_marker(None) is None


def test_a_resolved_marker_announces_the_split_in_the_log():
    """Observed firing, not inferred from the wiring (non-negotiable 16).

    The line goes through ``logging_service.log_info``, so ``caplog`` is blind
    to it; the service is spied instead.
    """
    mock = MagicMock()
    mock.metrics_service.save_images_batch = MagicMock(return_value=([], []))
    mock._slice_to_target_contrast = lambda pred, target: (pred, target)
    mock._slice_to_target_contrast_single = lambda ksp: ksp
    mock._resolve_federated_target_start = (
        DiffusionTrainingStrategy._resolve_federated_target_start
    )
    stack = torch.zeros(2, 16, 8, 8)

    DiffusionTrainingStrategy._log_validation_images_to_tensorboard(
        mock,
        stack,
        stack.clone(),
        stack.clone(),
        {},
        batch_idx=0,
        is_image_domain=False,
        federated_target_channel_start=torch.tensor(8, dtype=torch.long),
    )

    assert any(
        "Federated split active" in str(c.args[0])
        for c in mock.logging_service.log_info.call_args_list
        if c.args
    )


# ─── Direct unit tests of the marker-validation helper.
#
# These used to load diffusion.py via ``spec_from_file_location`` under the
# synthetic name ``_diffusion_strategy_for_test``. That gives the module no
# package, so every relative import inside it raised "attempted relative import
# with no known parent package" — and the blanket ``except Exception:
# pytest.skip`` turned a broken loader into seven green-looking skips, so the
# helper below went untested for as long as the loader was broken. The module
# imports fine by its real name (two sibling test files already do it), which
# also makes the "transitive imports are heavy" rationale moot.


@pytest.fixture(scope="module")
def Strategy():
    return DiffusionTrainingStrategy


def test_resolve_federated_target_start_returns_none_when_marker_absent(Strategy):
    """``None`` marker → render full stack (historical behaviour preserved)."""
    preds = torch.zeros(2, 16, 8, 8)
    tgts = torch.zeros(2, 16, 8, 8)
    assert Strategy._resolve_federated_target_start(None, preds, tgts) is None


def test_resolve_federated_target_start_accepts_int(Strategy):
    """A scalar int marker (test-fixture style) is honoured."""
    preds = torch.zeros(1, 16, 8, 8)
    tgts = torch.zeros(1, 16, 8, 8)
    assert Strategy._resolve_federated_target_start(8, preds, tgts) == 8


def test_resolve_federated_target_start_accepts_0d_tensor(Strategy):
    """The dataset publishes the marker as a 0-d long tensor."""
    preds = torch.zeros(1, 16, 8, 8)
    tgts = torch.zeros(1, 16, 8, 8)
    marker = torch.tensor(8, dtype=torch.long)
    assert Strategy._resolve_federated_target_start(marker, preds, tgts) == 8


def test_resolve_federated_target_start_accepts_stacked_uniform_batch(Strategy):
    """All entries in a stacked batch agree → return the common value."""
    preds = torch.zeros(4, 16, 8, 8)
    tgts = torch.zeros(4, 16, 8, 8)
    marker = torch.tensor([8, 8, 8, 8], dtype=torch.long)
    assert Strategy._resolve_federated_target_start(marker, preds, tgts) == 8


def test_resolve_federated_target_start_returns_none_on_mixed_batch(Strategy):
    """Mixed federated / single-contrast batch can't be sliced uniformly.

    Returning ``None`` falls back to the historical full-stack render —
    safer than silently slicing only some samples.
    """
    preds = torch.zeros(2, 16, 8, 8)
    tgts = torch.zeros(2, 16, 8, 8)
    marker = torch.tensor([8, 0], dtype=torch.long)
    assert Strategy._resolve_federated_target_start(marker, preds, tgts) is None


def test_resolve_federated_target_start_rejects_out_of_bounds(Strategy):
    """A stale marker pointing past tensor shape is treated as absent."""
    preds = torch.zeros(1, 16, 8, 8)
    tgts = torch.zeros(1, 16, 8, 8)
    assert Strategy._resolve_federated_target_start(99, preds, tgts) is None
    assert Strategy._resolve_federated_target_start(0, preds, tgts) is None
    assert Strategy._resolve_federated_target_start(-1, preds, tgts) is None


def test_resolve_federated_target_start_rejects_shape_mismatch(Strategy):
    """If predictions and targets disagree on channel count, the marker is
    only honoured when the boundary fits BOTH — otherwise None."""
    preds = torch.zeros(1, 16, 8, 8)
    tgts = torch.zeros(1, 4, 8, 8)  # smaller, e.g. magnitude target
    assert Strategy._resolve_federated_target_start(8, preds, tgts) is None
