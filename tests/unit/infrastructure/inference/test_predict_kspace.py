"""Predict runs the arm's k-space normalisation and its J-sample estimator (#2581, #2329).

Pinned, each with the violation that turns it red:

* predict's output equals validation's ``RecorruptedInference`` mean for the same
  seed and batch key, on an M4Raw-shaped volume with a stub network, for ``r2r``
  (arm A) and ``nullspace_r2r`` (arm G), J = 5;
* the scale predict divides by is the one ``KSpaceNormalizationTransform``
  computes, and the output comes back in scanner units;
* planted: predict without the wrapper (one forward) differs;
* a strategy that normalises itself is not normalised a second time;
* the self-calibrated ``Sigma_n`` comes from the loader's own fit function.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spectramr.config.settings import TrainingSettings
from spectramr.core.metrics.coil_manifold import as_interleaved
from spectramr.infrastructure.inference.predict_kspace import PredictKSpace
from spectramr.infrastructure.inference.reconstruction_inference_strategy import (
    ReconstructionInferenceStrategy,
)
from spectramr.infrastructure.training.strategies.mixins.recorrupted_inference import (
    RecorruptedInference,
)
from spectramr.pipelines import infer as infer_mod

tio = pytest.importorskip("torchio")

_ARMS = Path("experiments/inprogress/n2n")
_S, _C, _H, _W = 3, 4, 32, 32
_FILE = "2022061003_T101"


def _same(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Equal up to float32 rounding of the scanner-unit output (predict divides the
    complex tensor, the loader its interleaved copy)."""
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6 * float(expected.abs().max()))


class _Stub(nn.Module):
    """Nonlinear, so the mean of J forwards is not the forward of the mean."""

    in_channels = 2 * _C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) * 2.0


def _arm(name: str, **data) -> TrainingSettings:
    """The arm on the group route unless ``data`` says otherwise.

    The comparisons below build one whole-volume subject, which is what the
    group route serves; the arms themselves declare the slice route (#2673),
    which ``test_the_arms_route_scales_each_slice`` covers.
    """
    cfg = TrainingSettings.from_yaml(_ARMS / f"{name}.yaml")
    data = {"slice_level_records": False, **data}
    return cfg.model_copy(update={"data": cfg.data.model_copy(update=data)})


def _volume(seed: int = 0) -> torch.Tensor:
    """Raw-unit complex ``(S, C, H, W)``, scanner-sized, with unacquired columns."""
    gen = torch.Generator().manual_seed(seed)
    shape = (_S, _C, _H, _W)
    y = torch.complex(torch.randn(shape, generator=gen), torch.randn(shape, generator=gen)) * 800
    y[..., _W - 6 :] = 0
    return y


def _maps() -> torch.Tensor:
    gen = torch.Generator().manual_seed(5)
    shape = (_S, _C, _H, _W)
    m = torch.complex(torch.randn(shape, generator=gen), torch.randn(shape, generator=gen))
    return m / m.abs().pow(2).sum(dim=1, keepdim=True).sqrt()


def _write_maps(root: Path, **meta_update) -> None:
    from spectramr.data.datasets.m4raw_coil_maps import MAP_FORMAT_VERSION, map_path, save_coil_maps
    from spectramr.infrastructure.physics.scan_coil_calibration import DEFAULT_CALIBRATION

    meta = {"format_version": MAP_FORMAT_VERSION, **DEFAULT_CALIBRATION, **meta_update}
    save_coil_maps(map_path(root, _FILE), _maps(), meta)


def _training_subject(raw: torch.Tensor):
    """What the loader serves at validation, after the training transform."""
    from spectramr.data.transforms.normalization import KSpaceNormalizationTransform

    stacked = as_interleaved(raw).permute(1, 2, 3, 0)  # (2C, H, W, S)
    subject = tio.Subject(
        input=tio.ScalarImage(tensor=stacked.clone()),
        target=tio.ScalarImage(tensor=stacked.clone()),
        kspace=tio.ScalarImage(tensor=stacked.clone()),
    )
    return KSpaceNormalizationTransform(percentile=0.99, scale_domain="image")(subject)


def _predict(cfg, raw: torch.Tensor, tmp_path: Path, monkeypatch, kspace=None) -> torch.Tensor:
    """The production ``_process_single_file`` on one file, output captured unsaved."""
    saved: list[torch.Tensor] = []
    model = _Stub()
    strategy = ReconstructionInferenceStrategy(model, torch.device("cpu"), {})
    monkeypatch.setattr(infer_mod, "_load_input", lambda f, d: raw.clone())
    monkeypatch.setattr(infer_mod, "_save_output", lambda out, *a, **k: saved.append(out))
    infer_mod._process_single_file(
        tmp_path / f"{_FILE}.h5",
        tmp_path,
        model,
        strategy,
        torch.device("cpu"),
        cfg,
        batch_size=_S,
        results={"outputs": [], "num_processed": 0},
        kspace=PredictKSpace(cfg, strategy) if kspace is None else kspace,
    )
    return saved[0]


def _validation(cfg, raw: torch.Tensor, extra: dict | None = None) -> torch.Tensor:
    """Validation's estimator on the same scan, de-normalised by the loader's scale."""
    from spectramr.infrastructure.training.strategies.mixins.recorrupted_inference import (
        resolve_recorrupted_inference,
    )

    subject = _training_subject(raw)
    scale = subject["kspace_scale"].reshape(1)
    batch = {"kspace_scale": scale, **(extra or {})}
    estimator = RecorruptedInference(resolve_recorrupted_inference(cfg))
    x = subject["input"].data.unsqueeze(0)  # (1, 2C, H, W, S), as the loop hands it
    out = estimator.predict(_Stub(), x, batch, batch_key=0)
    return out[0].permute(3, 0, 1, 2) * scale


@pytest.fixture(scope="module")
def arm_a() -> TrainingSettings:
    return _arm("n2n_a_r2r_single_ex_m4raw")


class TestPredictIsValidationsEstimator:
    def test_r2r_matches_validation_for_the_same_seed(self, arm_a, tmp_path, monkeypatch):
        raw = _volume()
        _same(_predict(arm_a, raw, tmp_path, monkeypatch), _validation(arm_a, raw))

    def test_nullspace_r2r_matches_validation_for_the_same_seed(self, tmp_path, monkeypatch):
        # Arm G reads maps from the input repetition alone (its declared recipe).
        _write_maps(tmp_path / "maps", recipe="input_repetition")
        cfg = _arm("n2n_g_nullspace_r2r_m4raw", coil_maps_root=str(tmp_path / "maps"))
        raw = _volume(1)
        maps = _maps().permute(1, 2, 3, 0).unsqueeze(0)  # the loader's (1, C, H, W, S)
        expected = _validation(cfg, raw, {"sensitivity_complex": maps})
        _same(_predict(cfg, raw, tmp_path, monkeypatch), expected)

    def test_predict_reads_the_store_the_arm_declares(self, tmp_path, monkeypatch):
        """A store written under a declared recipe and ACS width serves predict too (#2688)."""
        _write_maps(tmp_path / "maps", recipe="input_repetition", acs_size=20)
        cfg = _arm(
            "n2n_g_nullspace_r2r_m4raw",
            coil_maps_root=str(tmp_path / "maps"),
            coil_maps_recipe="input_repetition",
            coil_maps_acs_size=20,
        )
        raw = _volume(1)
        maps = _maps().permute(1, 2, 3, 0).unsqueeze(0)
        expected = _validation(cfg, raw, {"sensitivity_complex": maps})
        _same(_predict(cfg, raw, tmp_path, monkeypatch), expected)

    def test_planted_default_source_refuses_that_store(self, tmp_path):
        """Predict's builder before #2688: default calibration and recipe."""
        from spectramr.data.datasets.m4raw_coil_map_source import CoilMapSource
        from spectramr.infrastructure.physics.scan_coil_calibration import DEFAULT_CALIBRATION

        _write_maps(tmp_path / "maps", recipe="input_repetition", acs_size=20)
        planted = CoilMapSource(tmp_path / "maps", "raise", dict(DEFAULT_CALIBRATION))
        with pytest.raises(ValueError, match="estimated with recipe 'input_repetition'"):
            planted.maps_for(_FILE, [as_interleaved(_volume(1))])

    def test_planted_one_forward_is_not_the_estimator(self, arm_a, tmp_path, monkeypatch):
        """Predict as it was before #2329's predict half: one forward on the input."""
        raw = _volume()
        strategy = SimpleNamespace(owns_kspace_normalization=False)
        unwrapped = PredictKSpace(arm_a, strategy)
        unwrapped.spec = unwrapped.estimator = None
        planted = _predict(arm_a, raw, tmp_path, monkeypatch, kspace=unwrapped)
        with pytest.raises(AssertionError, match="not close"):
            _same(planted, _validation(arm_a, raw))


class TestNormalisation:
    def test_the_scale_is_the_training_transforms(self, arm_a):
        raw = _volume()
        prepared = PredictKSpace(arm_a, SimpleNamespace()).prepare(raw, _FILE)
        expected = float(_training_subject(raw)["kspace_scale"])
        assert torch.allclose(prepared.scale, torch.full((_S,), expected))
        torch.testing.assert_close(prepared.tensor, as_interleaved(raw / expected))
        assert float(prepared.tensor.abs().max()) < float(raw.abs().max()) / 10

    def test_the_arms_route_scales_each_slice(self):
        """As declared, the arm reads one slice per record, so the loader's scale is
        one slice's and predict measures each slice the same way."""
        cfg = TrainingSettings.from_yaml(_ARMS / "n2n_a_r2r_single_ex_m4raw.yaml")
        assert cfg.data.slice_level_records is True
        raw = _volume()
        prepared = PredictKSpace(cfg, SimpleNamespace()).prepare(raw, _FILE)
        expected = [float(_training_subject(raw[s : s + 1])["kspace_scale"]) for s in range(_S)]
        torch.testing.assert_close(prepared.scale, torch.tensor(expected))
        assert (
            PredictKSpace(cfg, SimpleNamespace()).provenance()["kspace_scale_granularity"]
            == "slice"
        )

    def test_log_compression_round_trips_through_the_interleaved_layout(self, arm_a):
        """Compressed as complex, decompressed as interleaved real: the pairing must agree."""
        processing = arm_a.data.processing.model_copy(update={"enable_log_scaling": True})
        cfg = arm_a.model_copy(
            update={"data": arm_a.data.model_copy(update={"processing": processing})}
        )
        prep = PredictKSpace(cfg, SimpleNamespace())
        raw = _volume()
        prepared = prep.prepare(raw, _FILE)
        scale = float(prepared.scale[0])
        assert not torch.allclose(prepared.tensor, as_interleaved(raw / scale), rtol=1e-3)
        _same(prep.denormalize(prepared.tensor, prepared.scale), as_interleaved(raw))

    def test_the_output_returns_in_scanner_units(self, tmp_path, monkeypatch):
        cfg = _arm("n2n_b_rep_pair_m4raw")  # no estimator: one forward
        raw = _volume()
        scale = float(_training_subject(raw)["kspace_scale"])
        out = _predict(cfg, raw, tmp_path, monkeypatch)
        _same(out, _Stub()(as_interleaved(raw / scale)) * scale)

    def test_a_strategy_that_normalises_itself_is_left_alone(self, arm_a):
        cfg = _arm("n2n_b_rep_pair_m4raw")
        owner = SimpleNamespace(owns_kspace_normalization=True)
        prep = PredictKSpace(cfg, owner)
        raw = _volume()
        assert prep.norm is None and prep.provenance()["kspace_normalised"] is False
        assert torch.equal(prep.prepare(raw, _FILE).tensor, raw)
        with pytest.raises(ValueError, match="runs its own sampler"):
            PredictKSpace(arm_a, owner)

    def test_the_provenance_records_the_spec(self, arm_a):
        stamp = PredictKSpace(arm_a, SimpleNamespace()).provenance()
        assert stamp["kspace_normalised"] is True
        assert stamp["kspace_scale_source"] == "input"
        assert stamp["kspace_scale_granularity"] == "volume"
        assert stamp["recorrupted_inference"]["samples"] == 5
        assert stamp["recorrupted_inference"]["target_mode"] == "r2r"


def test_a_self_calibrated_sigma_comes_from_the_loaders_fit(monkeypatch):
    import spectramr.data.datasets.m4raw_dataset as m4raw

    calls: list[tuple] = []

    def _fit(kspace, *, support_policy):
        calls.append((tuple(kspace.shape), support_policy))
        return torch.eye(_C, dtype=torch.complex64), 2

    monkeypatch.setattr(m4raw, "fit_scan_noise_covariance", _fit)
    cfg = _arm("n2n_e_selfcal_sigma_m4raw")
    prep = PredictKSpace(cfg, SimpleNamespace())
    prepared = prep.prepare(_volume(), _FILE)
    assert calls == [((_S, _C, _H, _W), cfg.data.noise_fit_support_policy)]
    sigma = prepared.batch_data(0, 2)["noise_covariance"]
    assert torch.equal(sigma, torch.eye(_C, dtype=torch.complex64))
    assert prep.provenance()["estimator_inputs"][_FILE]["noise_covariance"].startswith("self")
