"""Data-only probes of the self_supervised R4 arms on the m4raw route.

Every batch is built the way a run builds it -- the arm's YAML, the dataset, the
train/val transform chains, collation and ``BatchAdapter`` -- over a synthetic
three-repetition group written to h5 (``tests.fixtures.self_supervised_batches``).
No model is built. The grid is non-square (48 x 250) so a swapped H/W shows, and
the last 50 phase-encode columns are zero, as M4Raw's partial FOV is, so the
support is a strict subset of the axis.

Kept apart from ``test_self_supervised_cohort_consistency.py``, which reads the
YAMLs only; these load the data pipeline.
"""

from __future__ import annotations

import pytest
import torch

H, W, ZERO_FILL = 48, 250, 50
CENTER_FRACTION, ACCELERATION = 0.094, 4.0
#: M4Raw T1/T2/FLAIR share one 256x256 grid (docs/CLUSTER_DATA_LAYOUT.md);
#: re-measure per file with scripts/physics/measure_phase_encode_axis.py on the
#: cluster's `_nex3` manifests.
CORPUS_WIDTHS = (256,)
MAPS_ARM, VC_ARM = "ei_multicoil_m4raw_r4", "ei_unet_m4raw_r4"


def _complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0::2], x[:, 1::2])


@pytest.fixture(scope="module")
def maps_run():
    from tests.fixtures.self_supervised_batches import load_arm, real_batches_with_reps

    return real_batches_with_reps(load_arm(MAPS_ARM), shape=(H, W), pe_zero_fill=ZERO_FILL)


@pytest.fixture(scope="module")
def vc_run():
    from tests.fixtures.self_supervised_batches import load_arm, real_batches

    return real_batches(load_arm(VC_ARM), shape=(H, W), pe_zero_fill=ZERO_FILL)


def _support(reps: list[torch.Tensor]) -> torch.Tensor:
    return reps[0].abs().sum(dim=(0, 1, 2)) > 0


def _acs_band() -> slice:
    count = round(W * CENTER_FRACTION)
    return slice(W // 2 - count // 2, W // 2 - count // 2 + count)


def _mask_error(mask: torch.Tensor, support: torch.Tensor) -> str | None:
    """Why ``(B, 1, H, W)`` ``mask`` is not an R4 line mask on ``support``, or None."""
    if not torch.equal(mask, mask[..., :1, :].expand_as(mask)):
        return "the mask varies along dim -2: it undersamples the readout"
    lines = mask[:, 0, 0, :] > 0
    if (lines & ~support).any():
        return "the mask acquires zero-filled columns"
    expected = round(int(support.sum()) / ACCELERATION)
    if not (lines.sum(dim=-1) == expected).all():
        return f"{lines.sum(dim=-1).tolist()} lines, expected round(|support|/4) = {expected}"
    if not lines[:, _acs_band()].all():
        return "the ACS band is not inside the mask"
    return None


def test_the_input_mask_is_r4_on_the_acquired_support(maps_run) -> None:
    train, val, reps = maps_run
    support = _support(reps)
    assert int(support.sum()) == W - ZERO_FILL
    for batch in (train, val):
        assert batch.mask is not None and batch.mask.shape == (2, 1, H, W)
        assert _mask_error(batch.mask, support) is None, _mask_error(batch.mask, support)


@pytest.mark.parametrize(
    "plant",
    ["readout", "zero-filled", "r324", "no-acs"],
)
def test_the_mask_predicate_rejects_each_planted_shape(plant: str) -> None:
    support = torch.zeros(W, dtype=torch.bool)
    support[: W - ZERO_FILL] = True
    lines = torch.zeros(W, dtype=torch.bool)
    lines[_acs_band()] = True
    free = torch.nonzero(support & ~lines).flatten()
    lines[free[: round(int(support.sum()) / ACCELERATION) - int(lines.sum())]] = True
    mask = lines.float().expand(1, 1, H, W).clone()
    assert _mask_error(mask, support) is None
    if plant == "readout":
        mask[..., 0, :] = 0
    elif plant == "zero-filled":
        # The #2327(b) shape: a drawn line in the zero-filled columns.
        mask[..., free[0]] = 0
        mask[..., W - 1] = 1
    elif plant == "r324":
        # The #2382 count: R taken over the whole axis instead of the support.
        mask[..., free[-12:]] = 1
    else:
        mask[..., _acs_band().start] = 0
        mask[..., free[-1]] = 1
    assert _mask_error(mask, support) is not None


def test_the_measurement_is_the_masked_input(maps_run) -> None:
    train, val, _ = maps_run
    for batch in (train, val):
        measured = batch.metadata["measured_kspace"]
        assert torch.equal(measured, batch.input)
        assert (batch.input * (1 - batch.mask)).abs().max() == 0
        assert not torch.equal(batch.input, batch.target)


def test_the_maps_are_espirit_on_the_masked_input(maps_run) -> None:
    """The store (repetition 0, ACS 24) equals per-load ESPIRiT on the masked input."""
    import torchio as tio

    from spectramr.data.transforms.registry import build_transform

    train, _, _ = maps_run
    maps = train.metadata["sensitivity_complex"]
    assert torch.is_complex(maps) and maps.shape == (2, 4, H, W)
    subject = tio.Subject(input=tio.ScalarImage(tensor=train.input[0][..., None]))
    per_load = build_transform("espirit_sensitivity", acs_size=24, keys=["input"])(subject)
    expected = per_load["sensitivity_complex"].data.permute(3, 0, 1, 2)[0]
    torch.testing.assert_close(maps[0], expected, rtol=1e-4, atol=1e-5)


def test_the_validation_target_is_the_leave_one_out_mean(maps_run) -> None:
    """Of the repetitions registered onto the input (``nex_registration: translation``)."""
    from spectramr.data.datasets.m4raw_dataset import _average_reps
    from spectramr.data.datasets.m4raw_registration import estimate_shift

    _, val, reps = maps_run
    shifts = [(0.0, 0.0)] + [estimate_shift(reps[0], r) for r in reps[1:]]
    expected = _average_reps(reps, "phase_aligned_mean", exclude_index=0, shifts=shifts)[0]
    scale = val.metadata["kspace_scale"][0]
    served = _complex(val.target)[0] * scale
    torch.testing.assert_close(served, expected, rtol=1e-4, atol=1e-4)
    # Planted contrasts: the all-repetitions mean contains the input, and the
    # unregistered mean superposes the shifted repetitions.
    pooled = _average_reps(reps, "phase_aligned_mean", shifts=shifts)[0]
    assert not torch.allclose(served, pooled, rtol=1e-4, atol=1e-4)
    assert any(s != (0.0, 0.0) for s in shifts[1:]), "the fixture's repetitions do not move"
    unregistered = _average_reps(reps, "phase_aligned_mean", exclude_index=0)[0]
    assert not torch.allclose(served, unregistered, rtol=1e-4, atol=1e-4)


def test_the_multicoil_selector_is_emitted_on_this_batch(maps_run) -> None:
    """The arm's computer, its context-metric names and the EI adapter's F S x_hat
    yield a finite val_reference_noise_corrected_mse; without it there is no best
    checkpoint. No network: the prediction is a random coil-combined image."""
    from spectramr.core.metrics.computer import create_validation_metrics_computer
    from spectramr.infrastructure.training.strategies.equivariant_imaging_ops import (
        context_metric_kspace,
    )
    from spectramr.infrastructure.training.strategies.mixins.validation_context import (
        compute_context_metrics,
        context_metric_names,
    )
    from tests.fixtures.self_supervised_batches import load_arm

    config = load_arm(MAPS_ARM)
    _, val, _ = maps_run
    computer = create_validation_metrics_computer(
        list(config.validation.scoring.compute), device="cpu", domain="image", data_range=1.0
    )
    # The arm also scores SENSE columns and the repetition decomposition; this
    # probe is about the selector, which must be among the context metrics.
    assert "reference_noise_corrected_mse" in context_metric_names(computer)
    names = ("reference_noise_corrected_mse",)
    context = {"coil_sensitivities": val.metadata["sensitivity_complex"], "mask": val.mask}
    image = torch.randn(val.input.shape[0], 2, H, W, generator=torch.Generator().manual_seed(0))
    graded, domain = context_metric_kspace(image, context, True, False)
    assert graded.shape == val.target.shape and domain == "kspace"
    metrics = compute_context_metrics(
        computer, names, graded, val.target, val.input, val, config.model,
        lambda x: x, signal_domain=domain,
    )  # fmt: skip
    value = metrics["val_reference_noise_corrected_mse"]
    assert value == value and abs(value) != float("inf")


def test_the_virtual_coil_arm_carries_one_compressed_coil(vc_run) -> None:
    """``data.domain.target_channels: 2`` is what the batch carries after SVD."""
    from tests.fixtures.self_supervised_batches import load_arm

    train, val = vc_run
    declared = load_arm(VC_ARM).data.domain.target_channels
    for batch in (train, val):
        assert batch.input.shape[1] == batch.target.shape[1] == declared == 2
        assert torch.equal(batch.metadata["measured_kspace"], batch.input)
        assert "sensitivity_complex" not in batch.metadata


@pytest.mark.parametrize("width", CORPUS_WIDTHS)
def test_the_acs_band_covers_the_map_window_at_every_corpus_width(width: int) -> None:
    from tests.fixtures.self_supervised_batches import load_arm

    acs_size = load_arm(MAPS_ARM).data.coil_maps_acs_size
    assert round(width * CENTER_FRACTION) >= acs_size == 24
