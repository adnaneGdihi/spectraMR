"""Tests for the quality-matching orchestrator strategy."""

from __future__ import annotations

import importlib

import pytest
import torch
import yaml

from spectramr.infrastructure.physics.chain_fitter import FitResult
from spectramr.infrastructure.physics.degradation_chain import ChainLink, DegradationChain
from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
    QualityMatchingStrategy,
    require_quality_matching_config,
    write_calibration_artifact,
)


def _result(method: str = "differential_evolution") -> FitResult:
    return FitResult(
        chain=DegradationChain(links=(ChainLink(axis="complex_gaussian", theta=0.42),)),
        achieved={"tenengrad_variance": 0.0100},
        target={"tenengrad_variance": 0.0110},
        weights={"tenengrad_variance": 1.0},
        residual=0.001,
        initial_residual=0.01,
        gap_closed=0.9,
        n_evals=123,
        method=method,
        seed=5,
    )


# ── registration / wiring ─────────────────────────────────────────────


def test_strategy_is_registered():
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "quality_matching" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS


def test_registered_path_actually_resolves_to_a_class():
    """A registry entry pointing at a bad path is dead on arrival at runtime."""
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS["quality_matching"]
    module_path, _, cls_name = path.rpartition(".")
    module = importlib.import_module(module_path)
    assert hasattr(module, cls_name), f"{path} does not resolve to a class"
    assert isinstance(getattr(module, cls_name), type)


# ── the missing-block guard ───────────────────────────────────────────


def test_missing_config_block_raises_rather_than_defaulting():
    """A declared-optional block reads as None when absent; the strategy must RAISE.

    Silently substituting a default chain is the facade failure mode: the arm would
    advertise quality matching and run without it.
    """

    class _Training:
        quality_matching = None

    class _Config:
        training = _Training()

    with pytest.raises(ValueError, match=r"training\.quality_matching"):
        require_quality_matching_config(_Config())


def test_missing_training_section_also_raises():
    class _Config:
        training = None

    with pytest.raises(ValueError, match=r"training\.quality_matching"):
        require_quality_matching_config(_Config())


def test_present_config_block_is_returned():
    sentinel = object()

    class _Training:
        quality_matching = sentinel

    class _Config:
        training = _Training()

    assert require_quality_matching_config(_Config()) is sentinel


# ── the calibration artifact ──────────────────────────────────────────


def test_calibration_artifact_records_full_provenance(tmp_path):
    doc = yaml.safe_load(write_calibration_artifact(_result(), tmp_path).read_text())

    assert doc["chain"] == [{"axis": "complex_gaussian", "theta": 0.42}]
    fit = doc["fit"]
    assert fit["method"] == "differential_evolution"
    assert fit["seed"] == 5
    assert fit["n_evals"] == 123
    assert fit["gap_closed"] == pytest.approx(0.9)
    assert fit["residual"] == pytest.approx(0.001)
    assert fit["initial_residual"] == pytest.approx(0.01)


def test_calibration_artifact_reports_achieved_against_target_per_attribute():
    # A residual alone hides WHICH attribute missed. The per-attribute table is what
    # makes a fit auditable rather than a single number to trust.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        doc = yaml.safe_load(write_calibration_artifact(_result(), Path(d)).read_text())

    attr = doc["attributes"]["tenengrad_variance"]
    assert attr["target"] == pytest.approx(0.0110)
    assert attr["achieved"] == pytest.approx(0.0100)
    assert attr["weight"] == pytest.approx(1.0)


def test_calibration_artifact_emits_replayable_twin_config(tmp_path):
    """The artifact must carry config the production twin can consume verbatim."""
    doc = yaml.safe_load(write_calibration_artifact(_result(), tmp_path).read_text())

    twin = doc["digital_twin"]
    assert twin["progressive_degradations"] == ["complex_gaussian"]
    assert list(twin["degradation_ranges"]["complex_gaussian"]) == [0.42, 0.42]


def test_emitted_twin_config_is_accepted_by_the_real_simulator(tmp_path):
    """End-to-end: the artifact's digital_twin block must CONSTRUCT a simulator.

    A YAML block that looks right but is rejected at simulator construction would
    make every downstream arm fail at startup.
    """
    from spectramr.infrastructure.physics.digital_twin_simulator import (
        DigitalTwinSimulator,
    )

    doc = yaml.safe_load(write_calibration_artifact(_result(), tmp_path).read_text())
    twin = doc["digital_twin"]

    sim = DigitalTwinSimulator(
        im_size=(32, 32),
        progressive_degradations=twin["progressive_degradations"],
        degradation_ranges={k: tuple(v) for k, v in twin["degradation_ranges"].items()},
    )
    assert sim.degradation_ranges["complex_gaussian"] == (0.42, 0.42)


def test_calibration_artifact_records_spacing_when_supplied(tmp_path):
    path = write_calibration_artifact(_result(), tmp_path, spacing_mm=(5.0, 1.2, 1.2))
    doc = yaml.safe_load(path.read_text())
    assert doc["spacing_mm"] == [5.0, 1.2, 1.2]


def test_calibration_artifact_creates_a_missing_output_dir(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    path = write_calibration_artifact(_result(), nested)
    assert path.exists()


# ── target resolution (cohort vs literal) ─────────────────────────────


def _target_cfg(**kw):
    from spectramr.config.schemas.training.quality_matching import QualityTargetConfig

    base = {
        "source": "literal",
        "attributes": ["tenengrad_variance"],
        "override": {"tenengrad_variance": 0.01},
    }
    base.update(kw)
    return QualityTargetConfig(**base)


def test_literal_target_returns_the_declared_overrides():
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        resolve_target,
    )

    assert resolve_target(_target_cfg()) == {"tenengrad_variance": 0.01}


def test_cohort_target_is_the_median_over_the_cohort(monkeypatch):
    """source='cohort' must actually MEASURE, not fall back to a default.

    An advertised source that nothing reads is an unwired knob (pitfall #15), so
    this drives the real code path with a stubbed volume loader.
    """
    from spectramr.infrastructure.training.strategies import (
        quality_matching_strategy as qms,
    )

    measured = [
        {"tenengrad_variance": 0.010},
        {"tenengrad_variance": 0.020},
        {"tenengrad_variance": 0.060},
    ]
    calls = iter(measured)
    monkeypatch.setattr(qms, "_measure_volume", lambda path, attributes, *_: next(calls))

    got = qms.measure_cohort_target(
        [f"v{i}.h5" for i in range(3)], attributes=["tenengrad_variance"]
    )
    # Median, not mean: 0.020 rather than 0.030. A cohort has outliers.
    assert got["tenengrad_variance"] == pytest.approx(0.020)


def test_cohort_target_applies_overrides_on_top(monkeypatch):
    """The documented ablation pattern: inherit the cohort fit, pin one attribute."""
    from spectramr.infrastructure.training.strategies import (
        quality_matching_strategy as qms,
    )

    monkeypatch.setattr(
        qms,
        "measure_cohort_target",
        lambda paths, attributes, max_volumes=None, **_: {
            "tenengrad_variance": 0.05,
            "laplacian_variance": 0.9,
        },
    )
    cfg = _target_cfg(
        source="cohort",
        cohort_manifest="data/manifests/lq.json",
        attributes=["tenengrad_variance", "laplacian_variance"],
        override={"tenengrad_variance": 0.01},
    )
    got = qms.resolve_target(cfg, cohort_paths=["a.h5"])
    assert got["tenengrad_variance"] == pytest.approx(0.01)  # overridden
    assert got["laplacian_variance"] == pytest.approx(0.9)  # inherited


def test_cohort_target_without_paths_raises():
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        resolve_target,
    )

    cfg = _target_cfg(source="cohort", cohort_manifest="data/manifests/lq.json")
    with pytest.raises(ValueError, match="cohort"):
        resolve_target(cfg, cohort_paths=[])


def test_cohort_volume_paths_raises_on_an_unreadable_manifest(tmp_path):
    """Manifests are gitignored and regenerated on-cluster; a missing one must be LOUD.

    read_manifest_records is deliberately tolerant (returns None) for the audit's
    sake. Silently treating that as an empty cohort would fit a chain against
    nothing.
    """
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        cohort_volume_paths,
    )

    with pytest.raises(FileNotFoundError, match="manifest"):
        cohort_volume_paths(tmp_path / "absent.json")


def test_cohort_volume_paths_reads_a_real_manifest(tmp_path):
    import json

    manifest = tmp_path / "lq.json"
    manifest.write_text(json.dumps([{"path": "/data/a.h5"}, {"path": "/data/b.h5"}]))
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        cohort_volume_paths,
    )

    assert cohort_volume_paths(manifest) == ["/data/a.h5", "/data/b.h5"]


def test_the_registered_strategy_actually_calls_fit_chain():
    """MECHANISM-FIRES: the class must invoke the fitter, not merely validate config.

    This is the regression pin for the defect this strategy originally shipped with:
    it overrode only `_setup_strategy_specific_components`, so an arm declaring
    `training_mode: quality_matching` validated cleanly, logged its axes, then trained
    a vanilla reconstruction model and fitted NOTHING. A green smoke run looked
    identical to a working one -- exactly the failure mode of #647.

    Asserting on the source is deliberate: constructing the real strategy needs the
    full DI/model harness, and a test that cannot run is a test that does not guard.
    """
    import inspect

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        QualityMatchingStrategy,
        run_quality_matching,
    )

    # The orchestration function must really drive the fitter.
    assert "fit_chain(" in inspect.getsource(run_quality_matching)
    assert "write_calibration_artifact(" in inspect.getsource(run_quality_matching)

    # And the strategy must really call the orchestration, from a TRAINING hook --
    # setup alone runs before any data exists and could not fit anything.
    assert hasattr(QualityMatchingStrategy, "_compute_losses_impl"), (
        "the strategy overrides no training hook, so the fit can never run"
    )
    body = inspect.getsource(QualityMatchingStrategy._compute_losses_impl)
    assert "run_quality_matching(" in body


def test_run_quality_matching_fits_and_writes_for_a_literal_target(tmp_path):
    """End-to-end through the real orchestration, with no training harness."""
    import torch

    from spectramr.config.schemas.training.quality_matching import QualityMatchingConfig
    from spectramr.infrastructure.physics.degradation_chain import (
        ChainLink,
        DegradationChain,
    )
    from spectramr.infrastructure.physics.quality_descriptors import measure_attributes
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        run_quality_matching,
    )

    torch.manual_seed(0)
    hq = torch.rand(2, 32, 32)
    truth = DegradationChain(links=(ChainLink(axis="t2star_blur", theta=0.55),))
    target = measure_attributes(
        truth.apply(hq.unsqueeze(1), seed=3).squeeze(1).abs(),
        attributes=["tenengrad_variance"],
    )

    cfg = QualityMatchingConfig(
        axes=["t2star_blur"],
        target={
            "source": "literal",
            "attributes": ["tenengrad_variance"],
            "override": target,
        },
        max_evals=80,
        fit_seed=3,
        output_dir=str(tmp_path),
        # Quality-only: this test has no header, and match_spacing would rightly
        # refuse to guess a voxel size. Geometry is covered separately below.
        match_spacing=False,
        # No HQ manifest here either; synthesis is covered by its own tests.
        synthesise=False,
    )

    result = run_quality_matching(cfg, hq)

    assert result.chain.axes == ("t2star_blur",)
    assert result.gap_closed >= cfg.min_gap_closed
    # The artifact is the arm's deliverable; if it is absent the run produced nothing.
    assert (tmp_path / "calibration.yaml").exists()


def test_extract_hq_volume_raises_rather_than_guessing():
    import torch

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        extract_hq_volume,
    )

    t = torch.rand(2, 8, 8)
    assert extract_hq_volume(t) is t
    assert extract_hq_volume({"target": t}) is t
    # A batch with no recognisable volume must NOT fall back to some other tensor:
    # fitting the wrong tensor still yields a confident-looking calibration.
    with pytest.raises(ValueError, match="could not find a high-quality volume"):
        extract_hq_volume({"mask": t})
    with pytest.raises(ValueError, match="tensor or mapping"):
        extract_hq_volume(object())


def test_cohort_volume_paths_raises_when_no_record_carries_a_path(tmp_path):
    import json

    manifest = tmp_path / "lq.json"
    manifest.write_text(json.dumps([{"subject": "s1"}, {"subject": "s2"}]))
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        cohort_volume_paths,
    )

    with pytest.raises(ValueError, match="usable volume path"):
        cohort_volume_paths(manifest)


# ── the GEOMETRIC half: header -> imposed grid ────────────────────────


_HEADER = """<?xml version="1.0"?>
<ismrmrdHeader><encoding><reconSpace>
  <matrixSize><x>64</x><y>64</y><z>1</z></matrixSize>
  <fieldOfView_mm><x>64.0</x><y>64.0</y><z>5.0</z></fieldOfView_mm>
</reconSpace></encoding></ismrmrdHeader>"""


def test_match_spacing_raises_when_no_header_is_reachable():
    """No header on the batch AND no manifest -> RAISE, never assume 1 mm."""
    import torch

    from spectramr.config.schemas.training.quality_matching import QualityMatchingConfig
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        run_quality_matching,
    )

    cfg = QualityMatchingConfig(
        axes=["t2star_blur"],
        target={
            "source": "literal",
            "attributes": ["tenengrad_variance"],
            "override": {"tenengrad_variance": 0.01},
            "spacing_mm": (5.0, 2.0, 2.0),
        },
        match_spacing=True,
        synthesise=False,
    )
    with pytest.raises(ValueError, match="no ISMRMRD header is reachable"):
        run_quality_matching(cfg, torch.rand(1, 64, 64))


def test_match_spacing_imposes_the_grid_and_records_both_spacings(tmp_path):
    """MECHANISM-FIRES for the geometric half.

    The header says 1 mm in-plane; the target says 2 mm. The fit must run on a
    resampled volume and the artifact must record BOTH spacings -- no quality
    attribute reveals whether the grid was imposed.
    """
    import torch

    from spectramr.config.schemas.training.quality_matching import QualityMatchingConfig
    from spectramr.infrastructure.physics.degradation_chain import (
        ChainLink,
        DegradationChain,
    )
    from spectramr.infrastructure.physics.quality_descriptors import (
        measure_attributes,
        normalise_intensity,
        resample_to_spacing,
    )
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        run_quality_matching,
    )

    torch.manual_seed(0)
    hq = torch.rand(2, 64, 64)

    # Derive a REACHABLE target: what a mid-severity chain yields on the volume
    # AFTER the grid is imposed. Hardcoding a number risks an unreachable target,
    # which the DegenerateFitError guard would rightly refuse -- masking whether
    # the geometry fired.
    on_grid = resample_to_spacing(hq, (5.0, 1.0, 1.0), (5.0, 2.0, 2.0))
    # A literal target lives in the normalised unit: the fit volume is put at p99 = 1.
    on_grid = normalise_intensity(on_grid, 99.0)
    truth = DegradationChain(links=(ChainLink(axis="t2star_blur", theta=0.5),))
    reachable = measure_attributes(
        truth.apply(on_grid.unsqueeze(1), seed=1).squeeze(1).abs(),
        attributes=["tenengrad_variance"],
    )

    cfg = QualityMatchingConfig(
        axes=["t2star_blur"],
        target={
            "source": "literal",
            "attributes": ["tenengrad_variance"],
            "override": reachable,
            "spacing_mm": (5.0, 2.0, 2.0),
        },
        max_evals=60,
        fit_seed=1,
        match_spacing=True,
        synthesise=False,
        output_dir=str(tmp_path),
    )
    run_quality_matching(cfg, hq, hq_header=_HEADER)

    doc = yaml.safe_load((tmp_path / "calibration.yaml").read_text())
    assert doc["source_spacing_mm"] == [5.0, 1.0, 1.0]
    assert doc["spacing_mm"] == [5.0, 2.0, 2.0]


def test_extract_hq_header_returns_none_rather_than_raising():
    # Whether a header is REQUIRED is match_spacing's decision, not this helper's.
    import torch

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        extract_hq_header,
    )

    assert extract_hq_header({"target": torch.rand(1, 4, 4)}) is None
    assert extract_hq_header({"header": _HEADER}) == _HEADER
    assert extract_hq_header(torch.rand(1, 4, 4)) is None


# ── synthesis: turning a calibration into a dataset ───────────────────


def _fake_h5(path, n_slices=2, size=32):
    """A minimal fastMRI-shaped h5 with a real ISMRMRD header."""
    import h5py
    import numpy as np

    with h5py.File(path, "w") as f:
        rng = np.random.default_rng(0)
        f.create_dataset(
            "reconstruction_rss",
            data=rng.random((n_slices, size, size), dtype=np.float32),
        )
        f.create_dataset("ismrmrd_header", data=_HEADER.encode())
    return str(path)


def test_synthesise_pairs_writes_readable_pairs_and_a_v4_manifest(tmp_path):
    """MECHANISM-FIRES for synthesis: the arm's dataset must actually exist.

    Without this the fit is a number in a YAML: exp_qm_02a's data_root would point at
    a directory nothing produces, and the arm could not run.
    """
    import json

    from spectramr.infrastructure.physics.degradation_chain import (
        ChainLink,
        DegradationChain,
    )
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        synthesise_pairs,
    )

    src = [_fake_h5(tmp_path / f"vol{i}.h5") for i in range(3)]
    chain = DegradationChain(links=(ChainLink(axis="t2star_blur", theta=0.6),))

    manifest = synthesise_pairs(chain, src, tmp_path / "out", seed=1)
    doc = json.loads(manifest.read_text())

    assert doc["manifest_version"] == "4.0"
    assert len(doc["records"]) == 3
    assert doc["chain"] == [{"axis": "t2star_blur", "theta": 0.6}]

    rec = doc["records"][0]
    # The v4 fields IndexBuilder.load_paired_bids_manifest actually reads.
    assert {"primary_path", "target_path", "pairing_status", "split_hint"} <= set(rec)
    assert rec["pairing_status"] == "paired"
    from pathlib import Path

    assert Path(rec["primary_path"]).exists()
    assert Path(rec["target_path"]).exists()


def test_synthesised_manifest_loads_through_the_paired_loader(tmp_path):
    """exp_qm_02a reads this file through IndexBuilder, which accepts only '4.0'/'5.0'."""
    from types import SimpleNamespace

    from spectramr.config.schemas.data import DataPairingConfigSchema
    from spectramr.data.metadata.index_builder import IndexBuilder
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        synthesise_pairs,
    )

    src = [_fake_h5(tmp_path / f"vol{i}.h5") for i in range(3)]
    chain = DegradationChain(links=(ChainLink(axis="t2star_blur", theta=0.6),))
    manifest = synthesise_pairs(chain, src, tmp_path / "out", seed=1)

    config = SimpleNamespace(
        pairing=DataPairingConfigSchema(), split=SimpleNamespace(type="manifest")
    )
    train = IndexBuilder.load_paired_bids_manifest(manifest, "train", config)
    val = IndexBuilder.load_paired_bids_manifest(manifest, "val", config)
    assert len(train) + len(val) == 3


def test_synthesised_input_is_actually_degraded_relative_to_its_target(tmp_path):
    """The pair must differ. Writing the clean volume twice would train an identity."""
    import json

    from spectramr.core.metrics.registry import get_metric
    from spectramr.data.io_strategies import NiftiStrategy
    from spectramr.infrastructure.physics.degradation_chain import (
        ChainLink,
        DegradationChain,
    )
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        synthesise_pairs,
    )

    src = [_fake_h5(tmp_path / "v.h5")]
    chain = DegradationChain(links=(ChainLink(axis="t2star_blur", theta=0.9),))
    doc = json.loads(synthesise_pairs(chain, src, tmp_path / "out", seed=1).read_text())
    rec = doc["records"][0]

    lq = NiftiStrategy().load(rec["primary_path"])["data"].squeeze().unsqueeze(1)
    hq = NiftiStrategy().load(rec["target_path"])["data"].squeeze().unsqueeze(1)
    sharp = get_metric("tenengrad_variance")(hq)
    blurred = get_metric("tenengrad_variance")(lq)
    assert blurred < sharp, "the 'degraded' volume is not degraded"


def test_synthesis_uses_a_distinct_realisation_per_volume(tmp_path):
    """One artefact realisation reused across subjects teaches THAT artefact."""
    import json

    from spectramr.data.io_strategies import NiftiStrategy
    from spectramr.infrastructure.physics.degradation_chain import (
        ChainLink,
        DegradationChain,
    )
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        synthesise_pairs,
    )

    # Same source content in both volumes, so any difference is the realisation.
    src = [_fake_h5(tmp_path / "a.h5"), _fake_h5(tmp_path / "b.h5")]
    chain = DegradationChain(links=(ChainLink(axis="complex_gaussian", theta=0.8),))
    doc = json.loads(
        synthesise_pairs(chain, src, tmp_path / "out", seed=1, val_fraction=0.0).read_text()
    )
    a = NiftiStrategy().load(doc["records"][0]["primary_path"])["data"].squeeze()
    b = NiftiStrategy().load(doc["records"][1]["primary_path"])["data"].squeeze()
    assert not torch.equal(a, b)


def test_synthesise_pairs_rejects_an_empty_source_list(tmp_path):
    from spectramr.infrastructure.physics.degradation_chain import (
        ChainLink,
        DegradationChain,
    )
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        synthesise_pairs,
    )

    chain = DegradationChain(links=(ChainLink(axis="t2star_blur", theta=0.5),))
    with pytest.raises(ValueError, match="zero high-quality volumes"):
        synthesise_pairs(chain, [], tmp_path / "out", seed=1)


# ── the acquisition prior reaching the fit ────────────────────────────


_HEADER_LQ = """<?xml version="1.0"?>
<ismrmrdHeader><acquisitionSystemInformation>
  <systemFieldStrength_T>0.3</systemFieldStrength_T>
</acquisitionSystemInformation><encoding><reconSpace>
  <matrixSize><x>32</x><y>32</y><z>1</z></matrixSize>
  <fieldOfView_mm><x>64.0</x><y>64.0</y><z>5.0</z></fieldOfView_mm>
</reconSpace></encoding></ismrmrdHeader>"""

_HEADER_HQ_3T = """<?xml version="1.0"?>
<ismrmrdHeader><acquisitionSystemInformation>
  <systemFieldStrength_T>3.0</systemFieldStrength_T>
</acquisitionSystemInformation><encoding><reconSpace>
  <matrixSize><x>32</x><y>32</y><z>1</z></matrixSize>
  <fieldOfView_mm><x>32.0</x><y>32.0</y><z>5.0</z></fieldOfView_mm>
</reconSpace></encoding></ismrmrdHeader>"""


def _fake_h5_with(path, header, n_slices=2, size=32):
    import h5py
    import numpy as np

    with h5py.File(path, "w") as f:
        rng = np.random.default_rng(1)
        f.create_dataset(
            "reconstruction_rss",
            data=rng.random((n_slices, size, size), dtype=np.float32),
        )
        f.create_dataset("ismrmrd_header", data=header.encode())
    return str(path)


def test_acquisition_prior_reaches_the_fit_and_is_recorded(tmp_path):
    """MECHANISM-FIRES for the prior: it must reach theta0 AND be auditable.

    warm_start_theta and fit_chain's theta0 were both dead in the live path before
    this. Asserting the artifact records the prediction is what stops them going
    dead again silently.
    """
    import json

    import torch

    from spectramr.config.schemas.training.quality_matching import QualityMatchingConfig
    from spectramr.infrastructure.physics.degradation_chain import (
        ChainLink,
        DegradationChain,
    )
    from spectramr.infrastructure.physics.quality_descriptors import (
        measure_attributes,
        normalise_intensity,
    )
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        run_quality_matching,
    )

    lq = [_fake_h5_with(tmp_path / f"lq{i}.h5", _HEADER_LQ) for i in range(2)]
    lq_manifest = tmp_path / "lq.json"
    lq_manifest.write_text(json.dumps([{"path": p} for p in lq]))

    # A reachable target (a mid-noise chain on the normalised HQ), pinned by override:
    # the cohort still supplies the acquisition the prior reads. Random data alone would
    # saturate the single noise axis.
    hq = torch.rand(2, 32, 32)
    truth = DegradationChain(links=(ChainLink(axis="complex_gaussian", theta=0.5),))
    reachable = measure_attributes(
        truth.apply(normalise_intensity(hq, 99.0).unsqueeze(1), seed=2).squeeze(1).abs(),
        attributes=["tenengrad_variance"],
    )

    cfg = QualityMatchingConfig(
        axes=["complex_gaussian"],
        target={
            "source": "cohort",
            "cohort_manifest": str(lq_manifest),
            "attributes": ["tenengrad_variance"],
            "override": reachable,
        },
        max_evals=60,
        fit_seed=2,
        min_gap_closed=0.0,
        match_spacing=False,
        synthesise=False,
        output_dir=str(tmp_path / "out"),
    )
    assert cfg.acquisition_prior_enabled is True

    run_quality_matching(cfg, hq, hq_header=_HEADER_HQ_3T)

    doc = yaml.safe_load((tmp_path / "out" / "calibration.yaml").read_text())
    prior = doc["acquisition_prior"]
    assert prior["enabled"] is True
    # 3T -> 0.3T with neither cohort recording averages or bandwidth => -20 dB.
    assert prior["predicted_snr_delta_db"] == pytest.approx(-20.0)
    # complex_gaussian: 40 dB clean, -20 dB predicted => 20 dB => theta = 20/38.
    assert prior["theta0"][0] == pytest.approx(20.0 / 38.0, abs=1e-6)


def test_prior_raises_rather_than_guessing_a_field_strength(tmp_path):
    import json

    import torch

    from spectramr.config.schemas.training.quality_matching import QualityMatchingConfig
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        run_quality_matching,
    )

    lq = [_fake_h5_with(tmp_path / "lq.h5", _HEADER_LQ)]
    lq_manifest = tmp_path / "lq.json"
    lq_manifest.write_text(json.dumps([{"path": p} for p in lq]))

    cfg = QualityMatchingConfig(
        axes=["complex_gaussian"],
        target={
            "source": "cohort",
            "cohort_manifest": str(lq_manifest),
            "attributes": ["tenengrad_variance"],
            "override": {},
        },
        max_evals=20,
        match_spacing=False,
        synthesise=False,
        output_dir=str(tmp_path / "out"),
    )
    # No HQ header and no HQ manifest => the prior cannot be computed.
    with pytest.raises(ValueError, match="no high-quality acquisition"):
        run_quality_matching(cfg, torch.rand(1, 32, 32))


# ── multi-field selection: paired vs unpaired ─────────────────────────


def _mrix_manifest(tmp_path, *, paired: bool):
    """A MRIxFields-shaped manifest. paired=True gives travelling volunteers."""
    import json

    recs = []
    if paired:
        # Same subject at BOTH fields — Training_prospective.
        for sid in ("s1", "s2", "s3"):
            for field in (3.0, 0.1):
                recs.append(
                    {
                        "primary_path": f"/d/{sid}_{field}.nii.gz",
                        "field_strength": field,
                        "contrast": "t1w",
                        "subject_id": sid,
                        "pairing_group": f"{sid}|t1w",
                    }
                )
    else:
        # One field per volunteer — Training_retrospective.
        for i, field in enumerate((3.0, 3.0, 0.1, 0.1)):
            recs.append(
                {
                    "primary_path": f"/d/r{i}.nii.gz",
                    "field_strength": field,
                    "contrast": "t1w",
                    "subject_id": f"r{i}",
                    "pairing_group": f"r{i}|t1w",
                }
            )
    p = tmp_path / ("paired.json" if paired else "unpaired.json")
    p.write_text(json.dumps(recs))
    return p


def test_select_field_returns_only_that_field(tmp_path):
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        select_field,
    )

    m = _mrix_manifest(tmp_path, paired=True)
    assert len(select_field(m, 0.1)) == 3
    assert all("0.1" in p for p in select_field(m, 0.1))


def test_select_field_raises_and_lists_what_is_available(tmp_path):
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        select_field,
    )

    m = _mrix_manifest(tmp_path, paired=True)
    # An empty cohort silently becomes "measure nothing"; the message must name the
    # fields that DO exist so the typo is obvious.
    with pytest.raises(ValueError, match=r"no volumes at 7\.0 T"):
        select_field(m, 7.0)
    with pytest.raises(ValueError, match=r"Fields present"):
        select_field(m, 7.0)


def test_select_field_pairs_matches_subjects_across_fields(tmp_path):
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        select_field_pairs,
    )

    pairs = select_field_pairs(_mrix_manifest(tmp_path, paired=True), 3.0, 0.1)
    assert len(pairs) == 3
    for hq, lq in pairs:
        # Same subject on both sides — that is the whole point of a paired cohort.
        assert hq.split("/")[-1].split("_")[0] == lq.split("/")[-1].split("_")[0]
        assert "3.0" in hq and "0.1" in lq


def test_retrospective_cohort_yields_no_pairs_and_says_why(tmp_path):
    """A one-field-per-volunteer cohort must NOT silently produce zero pairs.

    Mislabelling a retrospective cohort as paired is the easy mistake; the message
    has to name the fix.
    """
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        select_field_pairs,
    )

    with pytest.raises(ValueError, match="pairing: unpaired"):
        select_field_pairs(_mrix_manifest(tmp_path, paired=False), 3.0, 0.1)


def test_select_field_pairs_honours_the_contrast_filter(tmp_path):
    import json

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        select_field_pairs,
    )

    recs = []
    for contrast in ("t1w", "t2w"):
        for field in (3.0, 0.1):
            recs.append(
                {
                    "primary_path": f"/d/s1_{contrast}_{field}.nii.gz",
                    "field_strength": field,
                    "contrast": contrast,
                    "subject_id": "s1",
                    "pairing_group": f"s1|{contrast}",
                }
            )
    m = tmp_path / "mc.json"
    m.write_text(json.dumps(recs))

    pairs = select_field_pairs(m, 3.0, 0.1, contrast="t2w")
    assert len(pairs) == 1
    assert all("t2w" in p for p in pairs[0])


def test_paired_agreement_is_perfect_for_an_identical_volume():
    import torch

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        paired_agreement,
    )

    v = torch.rand(2, 16, 16)
    got = paired_agreement(v, v.clone(), metrics=("ssim",))
    assert got["ssim"] == pytest.approx(1.0, abs=1e-3)


def test_paired_agreement_falls_for_a_wrong_volume():
    import torch

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        paired_agreement,
    )

    torch.manual_seed(0)
    a = torch.rand(2, 16, 16)
    b = torch.rand(2, 16, 16)
    assert paired_agreement(a, b, metrics=("ssim",))["ssim"] < 0.5


# ── explicit pair layout: a ULF->HF restoration manifest ──────────────


def _ulf_manifest(tmp_path, *, with_unpaired=False):
    """v4 shape: input_path = ULF (model input), target_path = HF (what it produces)."""
    import json

    recs = [
        {
            "subject_id": f"sub-{i:02d}",
            "contrast": "t1w",
            "input_path": f"/d/sub-{i:02d}_acq-ulf.nii.gz",
            "target_path": f"/d/sub-{i:02d}_acq-hf.nii.gz",
        }
        for i in range(3)
    ]
    if with_unpaired:
        # allow_unpaired manifests carry ULF volumes with no HF partner.
        recs.append(
            {
                "subject_id": "sub-99",
                "contrast": "t1w",
                "input_path": "/d/sub-99_acq-ulf.nii.gz",
                "target_path": None,
            }
        )
    p = tmp_path / "ulf_paired_v4.json"
    p.write_text(json.dumps(recs))
    return p


def test_explicit_pairs_invert_the_restoration_naming(tmp_path):
    """THE TRAP: the HQ source is the manifest's TARGET, not its input.

    A ULF->HF manifest is written for restoration, so input_path is the ULF scan.
    Reading it as the source would degrade the ULF volume toward itself and still
    produce a confident calibration, because the attributes would already nearly
    match.
    """
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        select_explicit_pairs,
    )

    pairs = select_explicit_pairs(_ulf_manifest(tmp_path))
    assert len(pairs) == 3
    for hq, lq in pairs:
        assert "acq-hf" in hq, "the HIGH-quality source must be the manifest's target"
        assert "acq-ulf" in lq, "the LOW-quality target must be the manifest's input"


def test_explicit_pairs_drop_one_sided_records(tmp_path):
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        select_explicit_pairs,
    )

    # 3 complete + 1 ULF-only; the unpaired one cannot anchor a paired check.
    assert len(select_explicit_pairs(_ulf_manifest(tmp_path, with_unpaired=True))) == 3


def test_explicit_pairs_raise_when_nothing_is_complete(tmp_path):
    import json

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        select_explicit_pairs,
    )

    m = tmp_path / "all_unpaired.json"
    m.write_text(
        json.dumps([{"input_path": "/d/a.nii.gz", "target_path": None, "contrast": "t1w"}])
    )
    with pytest.raises(ValueError, match="pairing: unpaired"):
        select_explicit_pairs(m)


def test_explicit_pairs_honour_contrast(tmp_path):
    import json

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        select_explicit_pairs,
    )

    m = tmp_path / "mc.json"
    m.write_text(
        json.dumps(
            [
                {
                    "input_path": "/d/a_ulf.nii.gz",
                    "target_path": "/d/a_hf.nii.gz",
                    "contrast": "t1w",
                },
                {
                    "input_path": "/d/b_ulf.nii.gz",
                    "target_path": "/d/b_hf.nii.gz",
                    "contrast": "t2w",
                },
            ]
        )
    )
    pairs = select_explicit_pairs(m, contrast="t2w")
    assert len(pairs) == 1 and "b_" in pairs[0][0]


def test_contrast_filters_ignore_case(tmp_path):
    """Generators write T1w/T1W while the arms declare t1w; an exact compare selects nothing."""
    import json

    from spectramr.infrastructure.training.strategies.qm_selection import select_paths
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        select_explicit_pairs,
        select_field,
        select_field_pairs,
    )

    m = tmp_path / "mixed_case.json"
    m.write_text(
        json.dumps(
            [
                {
                    "subject_id": "s1",
                    "field_strength": 3.0,
                    "contrast": "T1w",
                    "primary_path": "/d/s1_3t.nii.gz",
                    "input_path": "/d/s1_ulf.nii.gz",
                    "target_path": "/d/s1_3t.nii.gz",
                },
                {
                    "subject_id": "s1",
                    "field_strength": 0.1,
                    "contrast": "T1W",
                    "primary_path": "/d/s1_01t.nii.gz",
                },
            ]
        )
    )
    assert select_field(m, 3.0, contrast="t1w") == ["/d/s1_3t.nii.gz"]
    assert select_paths(m, field_t=0.1, contrast="t1w") == ["/d/s1_01t.nii.gz"]
    assert select_field_pairs(m, 3.0, 0.1, contrast="t1w") == [
        ("/d/s1_3t.nii.gz", "/d/s1_01t.nii.gz")
    ]
    assert len(select_explicit_pairs(m, contrast="t1w")) == 1


def test_explicit_pairs_resolve_against_the_manifest_data_root(tmp_path):
    """v4 records are relative to data_root; a bare relative path resolves against the cwd."""
    import json

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        select_explicit_pairs,
    )

    m = tmp_path / "v4.json"
    m.write_text(
        json.dumps(
            {
                "data_root": "/data/iso",
                "records": [
                    {
                        "input_path": "sub-01/anat/ulf.nii.gz",
                        "target_path": "sub-01/anat/hf.nii.gz",
                    },
                    {"input_path": "/abs/ulf.nii.gz", "target_path": "/abs/hf.nii.gz"},
                ],
            }
        )
    )
    assert select_explicit_pairs(m) == [
        ("/data/iso/sub-01/anat/hf.nii.gz", "/data/iso/sub-01/anat/ulf.nii.gz"),
        ("/abs/hf.nii.gz", "/abs/ulf.nii.gz"),
    ]


def test_explicit_layout_needs_no_field_strengths():
    """field_keyed needs two fields; explicit reads them off the records."""
    from spectramr.config.schemas.training.quality_matching import QualityTargetConfig

    cfg = QualityTargetConfig(
        source="cohort",
        cohort_manifest="data/manifests/ulf_paired_brain_isotropic_v4.json",
        attributes=["tenengrad_variance"],
        override={},
        pairing="paired",
        pair_layout="explicit",
    )
    assert cfg.source_field_t is None and cfg.target_field_t is None


# ── manifest path resolution: relative_path + data_root ───────────────


def test_cluster_manifest_paths_resolve_against_data_root(tmp_path):
    """The cluster generators emit relative_path + a top-level data_root.

    A naive key sweep matches `filename` — which LOOKS like a path — and yields a
    bare basename that either fails to open or, far worse, opens a same-named file in
    the working directory. This is the shape regenerate_cluster_manifests.py and
    build_mrixfields2026_manifest.py actually write.
    """
    import json

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        cohort_volume_paths,
    )

    m = tmp_path / "cluster_v3.json"
    m.write_text(
        json.dumps(
            {
                "manifest_version": "3.0",
                "data_root": "databases/brain/fastmri/singlecoil_train",
                "records": [
                    {
                        "relative_path": "sub01/file_brain_AXT1_01.h5",
                        "filename": "file_brain_AXT1_01.h5",
                        "file_id": "file_brain_AXT1_01",
                    },
                    {
                        "relative_path": "sub02/file_brain_AXT1_02.h5",
                        "filename": "file_brain_AXT1_02.h5",
                        "file_id": "file_brain_AXT1_02",
                    },
                ],
            }
        )
    )
    got = cohort_volume_paths(m)
    assert got == [
        "databases/brain/fastmri/singlecoil_train/sub01/file_brain_AXT1_01.h5",
        "databases/brain/fastmri/singlecoil_train/sub02/file_brain_AXT1_02.h5",
    ]
    # The bare basename must NOT be what comes back.
    assert not any(p == "file_brain_AXT1_01.h5" for p in got)


def test_relative_record_without_data_root_raises(tmp_path):
    """Silently returning a basename is the dangerous outcome, so it must raise."""
    import json

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        cohort_volume_paths,
    )

    m = tmp_path / "rootless.json"
    m.write_text(json.dumps({"records": [{"relative_path": "a/b.h5"}]}))
    with pytest.raises(ValueError, match="no data_root"):
        cohort_volume_paths(m)


def test_absolute_keys_win_over_relative_ones(tmp_path):
    import json

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        cohort_volume_paths,
    )

    m = tmp_path / "mixed.json"
    m.write_text(
        json.dumps(
            {
                "data_root": "/should/not/be/used",
                "records": [{"primary_path": "/abs/a.nii.gz", "filename": "a.nii.gz"}],
            }
        )
    )
    assert cohort_volume_paths(m) == ["/abs/a.nii.gz"]


def test_field_selection_also_resolves_relative_paths(tmp_path):
    import json

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        select_field,
    )

    m = tmp_path / "mrix_v3.json"
    m.write_text(
        json.dumps(
            {
                "data_root": "databases/external/mrixfields2026",
                "records": [
                    {
                        "relative_path": "T1w/0.1T/v1.nii.gz",
                        "field_strength": 0.1,
                        "contrast": "t1w",
                        "subject_id": "v1",
                    },
                ],
            }
        )
    )
    assert select_field(m, 0.1) == ["databases/external/mrixfields2026/T1w/0.1T/v1.nii.gz"]


# ── the HQ manifest the fit falls back to for voxel geometry ──────────


class TestHqManifestReachesTheFit:
    """`hq_manifest` is read from the CANONICAL `data.source.index_path`.

    It used to be `getattr(self.config.data, "index_path", None)` — a
    string-keyed read of a name that folds to `data.source.index_path`, so it
    returned None for every arm. That did not degrade quietly. `match_spacing`
    resolves source spacing from a batch header first and this manifest second,
    and its own comment records that most datasets (``slice_dataset``, which
    serves ``dataset_type: kspace``) drop the ISMRMRD header during collation —
    so the manifest branch is the USUAL path. With it permanently None the arm
    fell through to ``raise ValueError`` telling it to "point data.index_path at
    a manifest", which it had already done. ``synthesise`` hit the same wall.

    Deliberately built from the REAL schema, not a duck-typed stub: a stub that
    carries ``index_path`` flat is a shape no real config produces, and agreeing
    with a dead reader is precisely how this survived.
    """

    def test_the_legacy_flat_spelling_still_reaches_the_canonical_path(self) -> None:
        from spectramr.config.schemas.data import DataConfigSchema

        cfg = DataConfigSchema(index_path="data/manifests/m4raw_train.json")
        assert cfg.source.index_path == "data/manifests/m4raw_train.json"

    def test_the_canonical_spelling_reaches_it_too(self) -> None:
        from spectramr.config.schemas.data import DataConfigSchema

        cfg = DataConfigSchema(source={"index_path": "data/manifests/x.json"})
        assert cfg.source.index_path == "data/manifests/x.json"

    def test_the_old_flat_read_would_have_returned_none(self) -> None:
        """Pins WHY the old expression was dead, so it cannot quietly return.

        `test_renames.py::TestNoStringKeyedReadsOfFoldedNames` fails if the
        `getattr` shape is reintroduced; this records what it cost.
        """
        from spectramr.config.schemas.data import DataConfigSchema

        cfg = DataConfigSchema(index_path="data/manifests/m4raw_train.json")
        assert getattr(cfg, "index_path", None) is None

    def test_an_undeclared_manifest_is_none_not_a_default(self) -> None:
        """Anti-vacuity: the field must be absent when the arm declares nothing,
        so the raise-when-unreachable branch can still fire."""
        from spectramr.config.schemas.data import DataConfigSchema

        assert DataConfigSchema().source.index_path is None


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: the hook returns self._zero_loss(): it computes no loss at all.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``ReconstructionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert QualityMatchingStrategy.__dict__["folds_image_losses"] is False
    assert QualityMatchingStrategy.__dict__["inline_losses"] == frozenset()


# ── NIfTI reference volumes and the declared HQ selection ─────────────


def _nifti_vol(path, shape_hwd=(4, 4, 2), zooms=(1.0, 1.0, 1.0)):
    import nibabel as nib
    import numpy as np

    rng = np.random.default_rng(3)
    arr = rng.random(shape_hwd, dtype=np.float32) + 0.1
    nib.save(nib.Nifti1Image(arr, np.diag([*zooms, 1.0])), str(path))
    return str(path)


def _qm_cfg(**kw):
    from types import SimpleNamespace

    base = {
        "fit_volumes": 1,
        "intensity_percentile": 99.0,
        "match_spacing": False,
        "target": SimpleNamespace(spacing_mm=None),
        **kw,
    }
    return SimpleNamespace(**base)


def test_fit_volumes_raises_when_fewer_volumes_are_selected(tmp_path):
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        _fit_volume,
    )

    one = _nifti_vol(tmp_path / "a.nii.gz")
    with pytest.raises(ValueError, match="fit_volumes=2 but only 1"):
        _fit_volume(_qm_cfg(fit_volumes=2), [one], None)


def test_fit_volumes_stacks_slices_of_equal_grids(tmp_path):
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        _fit_volume,
    )

    a = _nifti_vol(tmp_path / "a.nii.gz")
    b = _nifti_vol(tmp_path / "b.nii.gz")
    vol, _src, dst = _fit_volume(_qm_cfg(fit_volumes=2), [a, b], None)
    assert tuple(vol.shape) == (4, 4, 4)
    assert dst is None


def test_fit_volumes_raises_when_in_plane_shapes_disagree(tmp_path):
    # Planted: stacking a 4x4 and a 6x6 volume would fit against a silently
    # reshaped stack. It must refuse.
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        _fit_volume,
    )

    a = _nifti_vol(tmp_path / "a.nii.gz", shape_hwd=(4, 4, 2))
    b = _nifti_vol(tmp_path / "b.nii.gz", shape_hwd=(6, 6, 2))
    with pytest.raises(ValueError, match="disagree on in-plane shape"):
        _fit_volume(_qm_cfg(fit_volumes=2), [a, b], None)


def test_synthesis_of_nifti_sources_names_outputs_without_a_double_suffix(tmp_path):
    import json

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        synthesise_pairs,
    )

    src = [_nifti_vol(tmp_path / "subj_01.nii.gz")]
    chain = DegradationChain(links=(ChainLink(axis="t2star_blur", theta=0.6),))
    manifest = synthesise_pairs(chain, src, tmp_path / "out", seed=1)
    records = json.loads(manifest.read_text())["records"]
    assert records[0]["subject_id"] == "subj_01"
    assert records[0]["primary_path"].endswith("subj_01.nii.gz")
    assert ".nii.gz.nii.gz" not in records[0]["primary_path"]


def test_calibration_artifact_stamps_a_prior_the_arm_opted_out_of(tmp_path):
    import yaml

    out = write_calibration_artifact(
        _result(), tmp_path, acquisition_prior_enabled=False, theta0=None
    )
    doc = yaml.safe_load(out.read_text())
    assert doc["acquisition_prior"]["enabled"] is False
    assert doc["acquisition_prior"]["predicted_snr_delta_db"] is None


def test_explicit_nifti_pairs_run_end_to_end_and_write_a_manifest(tmp_path):
    """MECHANISM-FIRES for the NIfTI route: explicit pairs -> spacing -> fit -> synthesis.

    The HQ side is the pairs' TARGET (restoration naming inverted); no index_path is set.
    """
    import json

    from spectramr.config.schemas.training.quality_matching import QualityMatchingConfig
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        run_quality_matching,
    )

    hq = [_nifti_vol(tmp_path / f"hq{i}.nii.gz", (8, 8, 3), (1.0, 1.0, 2.0)) for i in range(2)]
    # LQ is the HQ plus noise, so its measured sharpness lies above a clean chain's
    # output and the fit has an interior optimum to find.
    import nibabel as nib
    import numpy as np

    lq = []
    for i in range(2):
        arr = nib.load(hq[i]).get_fdata(dtype=np.float32)
        noisy = arr + 0.3 * np.random.default_rng(i).standard_normal(arr.shape).astype(np.float32)
        out = tmp_path / f"lq{i}.nii.gz"
        nib.save(nib.Nifti1Image(noisy, np.diag([1.0, 1.0, 2.0, 1.0])), str(out))
        lq.append(str(out))
    manifest = tmp_path / "ulf.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": "4",
                "records": [
                    {"input_path": lq[i], "target_path": hq[i], "contrast": "t1w"} for i in range(2)
                ],
            }
        )
    )
    cfg = QualityMatchingConfig(
        axes=["complex_gaussian"],
        target={
            "source": "cohort",
            "cohort_manifest": str(manifest),
            "attributes": ["tenengrad_variance"],
            "pairing": "paired",
            "pair_layout": "explicit",
            "contrast": "t1w",
        },
        max_evals=10,
        min_gap_closed=0.0,
        match_spacing=False,
        enable_acquisition_prior=False,
        output_dir=str(tmp_path / "out"),
        synthesise=True,
    )
    result = run_quality_matching(cfg, torch.zeros(1), hq_manifest=None)
    assert result.chain.links[0].axis == "complex_gaussian"
    doc = yaml.safe_load((tmp_path / "out" / "calibration.yaml").read_text())
    assert doc["pairing"] == "paired"
    assert "paired_agreement_vs_real" in doc
    synth = json.loads(
        (tmp_path / "out" / "synthetic" / "quality_matching_synth_pairs.json").read_text()
    )
    assert {r["subject_id"] for r in synth["records"]} == {"hq0", "hq1"}


def test_nifti_cohort_spacing_reads_the_header_not_the_voxels(tmp_path, monkeypatch):
    # Performance guard: a cohort spacing needs the affine only. Decoding every
    # cohort volume (hundreds of ~200 MB MRIxFields volumes) just for voxel size is
    # the cost this avoids.
    from spectramr.data import reference_volume
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        _cohort_spacing,
    )

    path = _nifti_vol(tmp_path / "c.nii.gz", zooms=(1.0, 2.0, 3.0))

    def _no_decode(*_a, **_k):
        raise AssertionError("voxels were decoded for a spacing read")

    monkeypatch.setattr(reference_volume, "load_reference_volume", _no_decode)
    assert _cohort_spacing(path) == pytest.approx((3.0, 1.0, 2.0))


def _explicit_manifest(tmp_path, n=3):
    """A restoration-named manifest of ``n`` NIfTI pairs; LQ is noisy HQ."""
    import json

    import nibabel as nib
    import numpy as np

    manifest_records = []
    for i in range(n):
        hq = _nifti_vol(tmp_path / f"hq{i}.nii.gz", (8, 8, 3), (1.0, 1.0, 2.0))
        arr = nib.load(hq).get_fdata(dtype=np.float32)
        noisy = arr + 0.3 * np.random.default_rng(i).standard_normal(arr.shape).astype(np.float32)
        lq = tmp_path / f"lq{i}.nii.gz"
        nib.save(nib.Nifti1Image(noisy, np.diag([1.0, 1.0, 2.0, 1.0])), str(lq))
        manifest_records.append({"input_path": str(lq), "target_path": hq, "contrast": "t1w"})
    manifest = tmp_path / "ulf.json"
    manifest.write_text(json.dumps({"manifest_version": "4", "records": manifest_records}))
    return manifest


def _explicit_cfg(tmp_path, manifest, **kw):
    from spectramr.config.schemas.training.quality_matching import QualityMatchingConfig

    base = {
        "axes": ["complex_gaussian"],
        "target": {
            "source": "cohort",
            "cohort_manifest": str(manifest),
            "attributes": ["tenengrad_variance"],
            "pairing": "paired",
            "pair_layout": "explicit",
            "contrast": "t1w",
        },
        "max_evals": 10,
        "min_gap_closed": 0.0,
        "match_spacing": False,
        "enable_acquisition_prior": False,
        "output_dir": str(tmp_path / "out"),
        "synthesise": False,
    }
    base.update(kw)
    return QualityMatchingConfig(**base)


def test_held_out_pairs_score_agreement_and_the_undegraded_row_is_recorded(tmp_path):
    # Planted: the score must come from a subject the fit never saw, with the undegraded
    # comparator beside it. Three pairs at 1/3 hold-out: one scored, two fitted.
    import json

    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        run_quality_matching,
    )

    manifest = _explicit_manifest(tmp_path, n=3)
    cfg = _explicit_cfg(tmp_path, manifest, held_out_fraction=1 / 3)
    run_quality_matching(cfg, torch.zeros(1), hq_manifest=None)
    doc = yaml.safe_load((tmp_path / "out" / "calibration.yaml").read_text())
    assert doc["held_out"]["scope"] == "held_out"
    assert doc["held_out"]["fit_pairs"] == 2
    assert doc["held_out"]["scored_pairs"] == 1
    assert "paired_agreement_vs_real" in doc
    assert "paired_agreement_undegraded_vs_real" in doc
    assert json.dumps(doc["held_out"])  # plain, serialisable provenance


def test_seed_spread_is_mean_and_sample_sd_of_theta_over_seeds():
    # Planted: the estimand for the paired-vs-unpaired comparison. Three seeds at
    # theta 0.2, 0.4, 0.6 must give mean 0.4 and sd 0.2 exactly.
    from spectramr.infrastructure.physics.chain_fitter import FitResult
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        _theta_spread,
    )

    fits = [
        FitResult(
            chain=DegradationChain(links=(ChainLink(axis="complex_gaussian", theta=t),)),
            achieved={"tenengrad_variance": 1.0},
            target={"tenengrad_variance": 1.0},
            weights={"tenengrad_variance": 1.0},
            residual=0.01 * (i + 1),
            initial_residual=1.0,
            gap_closed=0.9,
            n_evals=10,
            method="differential_evolution",
            seed=i,
        )
        for i, t in enumerate((0.2, 0.4, 0.6))
    ]
    spread = _theta_spread(fits)
    assert spread["seeds"] == [0, 1, 2]
    assert spread["theta_mean"] == pytest.approx([0.4])
    assert spread["theta_sd"] == pytest.approx([0.2])


def test_fit_policy_is_stamped_so_the_artifact_shows_its_units(tmp_path):
    from spectramr.infrastructure.training.strategies.quality_matching_strategy import (
        run_quality_matching,
    )

    manifest = _explicit_manifest(tmp_path, n=2)
    cfg = _explicit_cfg(tmp_path, manifest, intensity_percentile=95.0, max_residual=5.0)
    run_quality_matching(cfg, torch.zeros(1), hq_manifest=None)
    policy = yaml.safe_load((tmp_path / "out" / "calibration.yaml").read_text())["fit_policy"]
    assert policy["intensity_percentile"] == 95.0
    assert policy["max_residual"] == 5.0
