"""Pretraining, saving and loading the FNO-EPG surrogate, and the refusal to use a random one.

Arm 51 built ``FNOEPGSurrogate()`` with its default random weights, loaded nothing, and put its
parameters in no optimiser: the "FNO-EPG" render was a frozen random network. The surrogate is
now fitted to the analytic EPG across the ranges the render meets, written with its measured
fidelity, loaded frozen, and ``use_fno_epg`` without weights raises.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fno_epg import FNOEPGSurrogate
from spectramr.infrastructure.physics.fno_epg_pretrain import (
    ECHO_TRAIN_LENGTH,
    build_fno_epg_surrogate,
    check_protocol_within_fit,
    fidelity,
    fit,
    load_fno_epg_surrogate,
    sample_training_set,
    save_fno_epg_surrogate,
)


def _tiny() -> FNOEPGSurrogate:
    torch.manual_seed(0)
    return FNOEPGSurrogate(tissue_channels=4, hidden_width=8, n_layers=1, max_etl=16)


class TestTrainingSet:
    def test_it_covers_the_ranges_the_render_meets(self) -> None:
        d = sample_training_set(4000, seed=1)
        assert d["t1"].min() >= 100.0 and d["t1"].max() <= 5000.0
        assert d["t2"].min() >= 10.0 and d["t2"].max() <= 2500.0
        assert (d["t2"] <= d["t1"]).all()  # transverse relaxation is never slower than T1
        # the generator's B1+ head is sigmoid + 0.5, so the fit covers (0.5, 1.5) end to end
        assert d["b1"].min() >= 0.5 and d["b1"].max() <= 1.5
        assert d["b1"].min() < 0.55 and d["b1"].max() > 1.45
        # the echo spacings TE / 8 gives for TE 10..150 ms: the protocol makes it a variable
        assert d["te_spacing"].min() >= 1.0 and d["te_spacing"].max() <= 20.0
        assert all(v.shape == (4000, 1, 1, 1) for v in d.values())

    def test_it_is_reproducible_and_the_target_is_a_finite_signal(self) -> None:
        a, b = sample_training_set(64, seed=3), sample_training_set(64, seed=3)
        assert all(torch.equal(a[k], b[k]) for k in a)
        assert torch.isfinite(a["target"]).all()
        assert a["target"].min() >= 0.0 and a["target"].max() <= 1.0 + 1e-4

    def test_the_target_is_the_analytic_signal_at_the_effective_echo(self) -> None:
        """Ideal CPMG: exp(-TE/T2) with TE = 8 * spacing, at b1 = 1."""
        d = sample_training_set(2000, seed=2)
        ideal = d["b1"].sub(1.0).abs() < 0.02
        want = torch.exp(-8.0 * d["te_spacing"] / d["t2"])
        assert torch.allclose(d["target"][ideal], want[ideal], atol=0.02)


class TestProtonDensityIsOutsideTheNetwork:
    """The normalised echo amplitude does not depend on proton density, which scales it.

    The surrogate is fitted at rho = 1. While rho was also one of the lift's four input
    channels, every rendered voxel (rho < 1) was out of the fitted distribution: a review
    measured the relative error growing from 0.03 at rho = 1 to about 0.14 at rho = 0.1.
    """

    @pytest.mark.parametrize("rho", [1.0, 0.7, 0.4, 0.1])
    def test_output_over_rho_is_the_same_at_every_rho_for_any_weights(self, rho: float) -> None:
        net = _tiny()
        for p in net.parameters():  # arbitrary weights: the property must not depend on fitting
            torch.nn.init.normal_(p, std=0.5)
        d = sample_training_set(32, seed=4)
        args = (d["t1"], d["t2"], d["b1"])
        at_one = net(torch.ones_like(d["t1"]), *args, te_spacing=d["te_spacing"])
        at_rho = net(torch.full_like(d["t1"], rho), *args, te_spacing=d["te_spacing"])
        assert torch.allclose(at_rho / rho, at_one, rtol=1e-5, atol=1e-7)


class TestFit:
    def test_the_loss_falls(self) -> None:
        net, data = _tiny(), sample_training_set(512, seed=0)
        losses = fit(net, data, steps=150, batch_size=256, lr=5e-3, seed=0)
        assert len(losses) == 150
        assert sum(losses[-10:]) / 10 < 0.5 * sum(losses[:10]) / 10

    def test_fidelity_improves_with_fitting_and_reports_its_measures(self) -> None:
        net, data = _tiny(), sample_training_set(512, seed=0)
        before = fidelity(net, data)
        fit(net, data, steps=150, batch_size=256, lr=5e-3, seed=0)
        after = fidelity(net, data)
        assert set(after) == {"mae", "rmse", "p99_abs", "max_abs"}
        assert after["mae"] < before["mae"]
        assert all(v >= 0.0 for v in after.values())


class TestSaveLoad:
    def _saved(self, tmp_path):
        net, data = _tiny(), sample_training_set(256, seed=0)
        fit(net, data, steps=20, batch_size=128, lr=5e-3, seed=0)
        path = tmp_path / "fno_epg.pt"
        save_fno_epg_surrogate(net, path, fidelity(net, data))
        return net, data, path

    def test_the_round_trip_reproduces_the_outputs_and_the_architecture(self, tmp_path) -> None:
        net, data, path = self._saved(tmp_path)
        loaded = load_fno_epg_surrogate(path, torch.device("cpu"))
        x = {k: v[:16] for k, v in data.items()}
        args = (x["rho"], x["t1"], x["t2"], x["b1"])
        net.eval()
        assert torch.allclose(
            net(*args, te_spacing=x["te_spacing"]), loaded(*args, te_spacing=x["te_spacing"])
        )
        assert loaded.lift[0].out_features == 8 and len(loaded.spectral_layers) == 1

    def test_a_loaded_surrogate_is_frozen_and_in_eval_mode(self, tmp_path) -> None:
        loaded = load_fno_epg_surrogate(self._saved(tmp_path)[2], torch.device("cpu"))
        assert not loaded.training
        assert not any(p.requires_grad for p in loaded.parameters())

    def test_a_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_fno_epg_surrogate(tmp_path / "absent.pt", torch.device("cpu"))

    def test_a_file_that_is_not_a_surrogate_raises(self, tmp_path) -> None:
        path = tmp_path / "other.pt"
        torch.save({"weights": torch.zeros(1)}, path)
        with pytest.raises(ValueError, match="not an FNO-EPG surrogate"):
            load_fno_epg_surrogate(path, torch.device("cpu"))


class TestSurrogateMatchesWhatTheRenderFeedsIt:
    def test_a_surrogate_fitted_for_another_train_length_is_refused(self, tmp_path) -> None:
        net, data = _tiny(), sample_training_set(64, seed=0)
        path = tmp_path / "w.pt"
        save_fno_epg_surrogate(net, path, fidelity(net, data))
        payload = torch.load(path, weights_only=True)
        payload["echo_train_length"] = ECHO_TRAIN_LENGTH + 4
        torch.save(payload, path)
        with pytest.raises(ValueError, match="echo train"):
            load_fno_epg_surrogate(path, torch.device("cpu"))

    def test_the_m4raw_protocol_lies_inside_the_fitted_spacings(self) -> None:
        check_protocol_within_fit({"T1": {"TE": 18.4}, "T2": {"TE": 128.0}, "FLAIR": {"TE": 98.0}})

    @pytest.mark.parametrize("te", [5.0, 200.0])
    def test_a_protocol_outside_the_fitted_spacings_is_refused(self, te: float) -> None:
        with pytest.raises(ValueError, match="outside the spacings"):
            check_protocol_within_fit({"X": {"TE": te}})


class TestBuild:
    def test_an_arm_that_does_not_use_it_gets_none(self) -> None:
        assert build_fno_epg_surrogate({}, torch.device("cpu")) is None

    def test_use_fno_epg_without_weights_raises_instead_of_building_a_random_one(self) -> None:
        with pytest.raises(ValueError, match="pretrain_fno_epg"):
            build_fno_epg_surrogate({"use_fno_epg": True}, torch.device("cpu"))

    def test_weights_for_an_arm_that_does_not_use_it_are_an_unread_knob(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="use_fno_epg"):
            build_fno_epg_surrogate(
                {"fno_epg_weights": str(tmp_path / "x.pt")}, torch.device("cpu")
            )

    def test_the_loaded_files_hash_is_logged(self, tmp_path, caplog) -> None:
        import hashlib
        import logging

        net, data = _tiny(), sample_training_set(64, seed=0)
        path = tmp_path / "w.pt"
        save_fno_epg_surrogate(net, path, fidelity(net, data))
        with caplog.at_level(logging.INFO):
            build_fno_epg_surrogate(
                {"use_fno_epg": True, "fno_epg_weights": str(path)}, torch.device("cpu")
            )
        assert hashlib.sha256(path.read_bytes()).hexdigest() in caplog.text

    def test_declared_weights_are_loaded_frozen(self, tmp_path) -> None:
        net, data = _tiny(), sample_training_set(128, seed=0)
        path = tmp_path / "w.pt"
        save_fno_epg_surrogate(net, path, fidelity(net, data))
        built = build_fno_epg_surrogate(
            {"use_fno_epg": True, "fno_epg_weights": str(path)}, torch.device("cpu")
        )
        assert built is not None and not any(p.requires_grad for p in built.parameters())
