"""``scripts/calibrate_rep_bv_estimator.py``: known references read as they must.

On rank-one coil images with iid complex noise and exact maps, the identity
estimator's NEX-equivalent is 1, its bias squared is zero and its repetition
variance equals the null-space floor; the coil-manifold projection's
NEX-equivalent is the coil count. Wrong maps are the planted violation: they push
the identity's bias squared off zero and the projection below the coil count,
which is exactly what the calibration exists to detect on real data.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from spectramr.core.metrics.coil_manifold import as_interleaved
from spectramr.data.batch_types import TrainingBatch
from spectramr.infrastructure.physics.fft_ops import fft2c

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "calibrate_rep_bv_estimator.py"
SIZE = 32
COILS = 4
SIGMA = 0.08
ESTIMATOR_NAMES = ("identity", "projection")


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("calibrate_rep_bv_estimator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def calibrate() -> Any:
    return _module()


def _maps(shift: int = 0) -> torch.Tensor:
    """Four smooth unit-norm maps, ``(C, H, W)`` complex; ``shift`` rolls them (wrong maps)."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, SIZE), torch.linspace(-1, 1, SIZE), indexing="ij")
    centres = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.5), (-1.0, 1.0, 1.0), (1.0, 1.0, 1.5)]
    maps = torch.stack(
        [
            torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) * torch.exp(1j * torch.tensor(p))
            for a, b, p in centres
        ]
    )
    maps = maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()
    return torch.roll(maps, shifts=(shift, shift), dims=(1, 2))


def _batch(seed: int, *, map_shift: int = 0) -> TrainingBatch:
    """Three repetitions of one slice; input rep 0, alt rep 1, target the mean of reps 1 and 2."""
    generator = torch.Generator().manual_seed(seed)
    maps = _maps()
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, SIZE), torch.linspace(-1, 1, SIZE), indexing="ij")
    anatomy = ((xx**2 + yy**2) < 0.6).float() * (1.0 + 0.3 * xx)
    clean = maps * anatomy
    reps = []
    for _ in range(3):
        noise = torch.randn(2, COILS, SIZE, SIZE, generator=generator) * (SIGMA / 2**0.5)
        reps.append(clean + torch.complex(noise[0], noise[1]))
    kspace = [as_interleaved(fft2c(r).unsqueeze(0)) for r in reps]
    target = (kspace[1] + kspace[2]) / 2
    return TrainingBatch(
        input=kspace[0],
        target=target,
        metadata={
            "input_alt": kspace[1],
            "sensitivity_complex": _maps(map_shift).unsqueeze(0),
            "subject_id": [f"synthetic_{seed}"],
            "slice_index": torch.tensor([seed]),
        },
    )


def _rows(calibrate: Any, seeds: range, **kwargs: Any) -> list[dict[str, Any]]:
    computer = calibrate.create_validation_metrics_computer(
        list(calibrate.METRICS), device="cpu", domain="kspace"
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.extend(calibrate.score_batch(_batch(seed, **kwargs), computer, torch.device("cpu")))
    return rows


def test_known_references_read_their_known_values(calibrate: Any) -> None:
    rows = _rows(calibrate, range(6))
    summary = calibrate.summarise(rows)
    identity = summary["estimators"]["identity"]
    projection = summary["estimators"]["projection"]
    assert summary["coils"] == COILS
    assert identity["nex_equivalent"]["mean"] == pytest.approx(1.0, abs=0.15)
    assert identity["rep_variance_over_floor"]["mean"] == pytest.approx(1.0, abs=0.2)
    assert abs(identity["bias_squared"]["mean"]) < 0.1 * identity["noise_floor"]["mean"]
    assert projection["nex_equivalent"]["mean"] == pytest.approx(COILS, rel=0.25)
    verdicts = summary["verdicts"]
    assert verdicts["identity_nex_equivalent"]["pass"] is True
    assert verdicts["identity_variance_over_floor"]["pass"] is True
    assert verdicts["projection_nex_equivalent"]["pass"] is True
    assert verdicts["identity_bias_squared"]["pass"] is None
    assert rows[0]["subject_id"] == "synthetic_0" and rows[0]["slice_index"] == "0"


def test_wrong_maps_are_detected_as_bias_and_a_projection_shortfall(calibrate: Any) -> None:
    """Rolled maps leak anatomy into the reference's null space, so the noise trace
    is over-estimated: the identity's bias squared leaves zero (here negative, the
    over-subtraction direction), its NEX-equivalent verdict fails, and the
    projection falls short of the coil count."""
    right = calibrate.summarise(_rows(calibrate, range(4)))
    wrong = calibrate.summarise(_rows(calibrate, range(4), map_shift=6))
    floor = right["estimators"]["identity"]["noise_floor"]["mean"]
    assert abs(wrong["estimators"]["identity"]["bias_squared"]["mean"]) > 3 * abs(
        right["estimators"]["identity"]["bias_squared"]["mean"]
    ) + 0.05 * abs(floor)
    assert wrong["verdicts"]["identity_nex_equivalent"]["pass"] is False
    assert (
        wrong["estimators"]["projection"]["nex_equivalent"]["mean"]
        < right["estimators"]["projection"]["nex_equivalent"]["mean"]
    )


def test_a_batch_without_maps_is_refused(calibrate: Any) -> None:
    batch = _batch(0)
    bare = TrainingBatch(
        input=batch.input, target=batch.target, metadata={"input_alt": batch.get("input_alt")}
    )
    computer = calibrate.create_validation_metrics_computer(
        list(calibrate.METRICS), device="cpu", domain="kspace"
    )
    with pytest.raises(ValueError, match="no coil maps"):
        calibrate.score_batch(bare, computer, torch.device("cpu"))


def _volume(seeds: range, *, zero_slice: int | None = None) -> TrainingBatch:
    """The slices of ``seeds`` stacked on a depth axis, one subject; one slice's maps zeroed."""
    slices = [_batch(seed) for seed in seeds]

    def stack(field: str) -> torch.Tensor:
        return torch.stack([s.get(field) for s in slices], dim=-1)

    maps = stack("sensitivity_complex")
    if zero_slice is not None:
        maps[..., zero_slice] = 0
    return TrainingBatch(
        input=torch.stack([s.input for s in slices], dim=-1),
        target=torch.stack([s.target for s in slices], dim=-1),
        metadata={
            "input_alt": stack("input_alt"),
            "sensitivity_complex": maps,
            "subject_id": ["synthetic_volume"],
        },
    )


def _computer(calibrate: Any) -> Any:
    return calibrate.create_validation_metrics_computer(
        list(calibrate.METRICS), device="cpu", domain="kspace"
    )


def test_a_volume_is_scored_slice_by_slice_with_its_identifiers(calibrate: Any) -> None:
    volume = _volume(range(3))
    rows = calibrate.score_loader([volume], _computer(calibrate), torch.device("cpu"), None)
    assert len(rows) == 3 * len(ESTIMATOR_NAMES)
    identity = [r for r in rows if r["estimator"] == "identity"]
    assert [r["slice_index"] for r in identity] == ["0", "1", "2"]
    assert {r["subject_id"] for r in identity} == {"synthetic_volume"}
    # The planted old path: identifiers counted against the folded B*D blank out.
    assert calibrate.identifier_strings(volume, 3)["subject_id"] == ""
    with pytest.raises(ValueError, match="expand_depth"):
        calibrate.score_batch(volume, _computer(calibrate), torch.device("cpu"))


def test_a_zero_map_slice_is_counted_and_the_inverse_mean_stays_finite(calibrate: Any) -> None:
    volume = _volume(range(4), zero_slice=2)
    rows = calibrate.score_loader([volume], _computer(calibrate), torch.device("cpu"), None)
    summary = calibrate.summarise(rows)
    for name in ESTIMATOR_NAMES:
        stats = summary["estimators"][name]
        assert stats["n"] == 4
        assert stats["n_zero_map"] == 1
        assert stats["nex_equivalent"]["n_nonfinite"] == 1
        assert stats["reference_noise_corrected_mse"]["n_nonfinite"] == 1
        inverse = stats["inverse_nex_equivalent"]
        assert inverse["n_nonfinite"] == 1 and inverse["n_slices"] == 3
        assert inverse["n_subjects"] == 1
        assert math.isfinite(inverse["slice_mean"]) and math.isfinite(inverse["subject_mean"])
        assert set(stats["nex_equivalent"]["quantiles"]) == {"p05", "p25", "p50", "p75", "p95"}
    identity = summary["estimators"]["identity"]["inverse_nex_equivalent"]
    assert identity["slice_mean"] == pytest.approx(1.0, abs=0.15)
    assert identity["n_unidentified"] == 0


def test_slices_without_a_subject_leave_the_subject_mean_undefined(calibrate: Any) -> None:
    rows = _rows(calibrate, range(2))
    for row in rows[:2]:
        row["subject_id"] = ""
    inverse = calibrate.summarise(rows)["estimators"]["identity"]["inverse_nex_equivalent"]
    assert inverse["n_unidentified"] == 1
    assert math.isfinite(inverse["slice_mean"]) and math.isnan(inverse["subject_mean"])


def test_the_limit_caps_loader_batches_not_slices(calibrate: Any) -> None:
    volumes = [_volume(range(2)), _volume(range(2, 4))]
    rows = calibrate.score_loader(volumes, _computer(calibrate), torch.device("cpu"), 1)
    assert len(rows) == 2 * len(ESTIMATOR_NAMES)


def test_outputs_land_as_csv_and_json(calibrate: Any, tmp_path: Path) -> None:
    rows = _rows(calibrate, range(2))
    summary = calibrate.summarise(rows)
    calibrate.write_outputs(tmp_path, rows, summary)
    assert (tmp_path / "calibration_rows.csv").read_text().count("\n") == len(rows) + 1
    assert '"verdicts"' in (tmp_path / "calibration_summary.json").read_text()
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        if torch.cuda.is_available():
            raise RuntimeError("CUDA is not available: skipped on a GPU box")
        calibrate._device("cuda")


def test_the_loader_is_built_at_the_validation_batch_size(calibrate: Any) -> None:
    """A row is one slice only at the arm's validation batch size of 1; the
    director's default is training's (2 on the n2n arms), which would average
    two slices into a row the per-slice reader refuses."""
    from types import SimpleNamespace

    settings = SimpleNamespace(
        validation=SimpleNamespace(loader=SimpleNamespace(batch_size=1)),
        data=SimpleNamespace(loader=SimpleNamespace(batch_size=2)),
    )
    assert calibrate.validation_loader_kwargs(settings) == {"num_workers": 0, "val_batch_size": 1}
    settings.validation.loader.batch_size = 3  # planted: it follows the arm, not a constant
    assert calibrate.validation_loader_kwargs(settings)["val_batch_size"] == 3


def test_estimator_rows_pair_with_an_arms_final_eval_keys(calibrate: Any) -> None:
    from spectramr.pipelines.test_pass import case_id

    volume = _volume(range(3))
    rows = calibrate.score_loader([volume], _computer(calibrate), torch.device("cpu"), None)
    by_estimator = calibrate.final_eval_by_estimator(rows)
    assert set(by_estimator) == set(ESTIMATOR_NAMES)
    identity_rows = [r for r in rows if r["estimator"] == "identity"]
    expected = {case_id(r) for r in identity_rows}
    assert set(by_estimator["identity"]["bias_squared"]) == expected
    assert len(expected) == 3  # one key per slice, not one per volume


def test_the_test_split_without_a_manifest_raises_rather_than_scoring_validation(
    calibrate: Any,
) -> None:
    from types import SimpleNamespace

    settings = SimpleNamespace(
        validation=SimpleNamespace(loader=SimpleNamespace(batch_size=1)),
        data=SimpleNamespace(source=SimpleNamespace(test_index_path=None)),
    )
    with pytest.raises(ValueError, match="No held-out test set"):
        calibrate.build_loader(settings, "test", torch.device("cpu"))
    with pytest.raises(ValueError, match="split must be"):
        calibrate.build_loader(settings, "train", torch.device("cpu"))
