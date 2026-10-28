"""``M4RawRepetitionDataset``: the null-space split and the alternate input.

The split's arithmetic is pinned in ``test_coil_split_r2r.py``; these pin the
dataset's half of the contract -- which modes branch where, what reaches the
subject, and what is refused at construction rather than at iteration 1.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

import spectramr.data.datasets.m4raw_dataset as m4raw_mod
from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset

_SHAPE = (2, 4, 32, 32)  # (S, C, H, W): 4 coils, like M4Raw
_UNSAMPLED = 6


def _write(path, *, seed: int, unsampled_cols: int = 0) -> None:
    """Smooth 4-coil anatomy plus noise, so ESPIRiT has something to calibrate."""
    rng = np.random.default_rng(seed)
    _, _, h, w = _SHAPE
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = np.stack([np.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres])
    anatomy = ((xx**2 + yy**2) < 0.5) * 20.0
    image = sens[None] * anatomy + rng.normal(size=_SHAPE) + 1j * rng.normal(size=_SHAPE)
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1))), axes=(-2, -1))
    kspace = (kspace / np.sqrt(h * w)).astype(np.complex64)
    if unsampled_cols:
        kspace[..., -unsampled_cols:] = 0
    with h5py.File(str(path), "w") as f:
        f.create_dataset("kspace", data=kspace)


def _dataset(tmp_path, *, reps=("01",), unsampled_cols=0, **kw) -> M4RawRepetitionDataset:
    files = []
    for i, rep in enumerate(reps):
        path = tmp_path / f"2022_T1{rep}.h5"
        _write(path, seed=i, unsampled_cols=unsampled_cols)
        files.append(path)
    kw.setdefault("coil_processing_mode", "none")
    return M4RawRepetitionDataset(h5_files=files, single_contrast=True, use_repetitions=True, **kw)


def _complex(stacked: torch.Tensor) -> torch.Tensor:
    """``(2C, H, W, S)`` interleaved -> ``(S, C, H, W)`` complex."""
    return torch.complex(stacked[0::2], stacked[1::2]).permute(3, 0, 1, 2)


def _stacked(kspace: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_complex`: ``(S, C, H, W)`` complex -> ``(2C, H, W, S)``."""
    out = torch.empty(2 * kspace.shape[1], *kspace.shape[2:], kspace.shape[0])
    out[0::2] = kspace.real.permute(1, 2, 3, 0)
    out[1::2] = kspace.imag.permute(1, 2, 3, 0)
    return out


class TestNullspaceR2R:
    def test_a_single_excitation_is_split_into_a_non_identity_pair(self, tmp_path) -> None:
        subject = _dataset(tmp_path, target_mode="nullspace_r2r")[0]
        assert subject["input"].data.shape == (8, 32, 32, 2)
        assert not torch.equal(subject["input"].data, subject["target"].data)

    def test_unsampled_phase_encode_columns_stay_exactly_zero(self, tmp_path) -> None:
        subject = _dataset(tmp_path, target_mode="nullspace_r2r", unsampled_cols=_UNSAMPLED)[0]
        for half in ("input", "target"):
            data = subject[half].data
            assert torch.count_nonzero(data[:, :, -_UNSAMPLED:, :]) == 0
            assert torch.count_nonzero(data[:, :, :-_UNSAMPLED, :]) > 0

    def test_leave_one_out_does_not_gate_it(self, tmp_path) -> None:
        """Shared with the val split, like r2r; the training split never averages."""
        ds = _dataset(tmp_path, target_mode="nullspace_r2r", nex_target_exclude_input=True)
        assert len(ds) == 1

    @pytest.mark.parametrize("whitening", ["none", "committed", "self_calibrated"])
    def test_every_whitening_source_serves(self, tmp_path, whitening) -> None:
        ds = _dataset(tmp_path, target_mode="nullspace_r2r", r2r_whitening=whitening)
        assert torch.isfinite(ds[0]["target"].data).all()
        assert bool(ds._r2r_cov_cache) == (whitening == "self_calibrated")

    def test_the_self_calibrated_whitener_is_fitted_once_per_scan(self, tmp_path, monkeypatch):
        import spectramr.infrastructure.physics.scan_coil_calibration as calibration

        calls = []
        real = calibration.fit_scan_covariance
        monkeypatch.setattr(
            calibration, "fit_scan_covariance", lambda *a, **k: calls.append(1) or real(*a, **k)
        )
        ds = _dataset(tmp_path, target_mode="nullspace_r2r", r2r_whitening="self_calibrated")
        _ = ds[0], ds[0]
        assert len(calls) == 1

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"r2r_whitening": "diagonal"}, "Unknown r2r_whitening"),
            ({"r2r_split_angle_deg": 90.0}, r"outside \(0, 90\)"),
            ({"r2r_split_angle_deg": 0.0}, r"outside \(0, 90\)"),
            ({"coil_processing_mode": "rss"}, "PHYSICAL coil vector"),
        ],
    )
    def test_an_unusable_declaration_raises_at_construction(self, tmp_path, kwargs, message):
        with pytest.raises(ValueError, match=message):
            _dataset(tmp_path, target_mode="nullspace_r2r", **kwargs)

    def test_average_reps_refuses_it(self) -> None:
        with pytest.raises(ValueError, match="construction site"):
            m4raw_mod._average_reps(
                [torch.zeros(1, 4, 4, 4, dtype=torch.complex64)], "nullspace_r2r"
            )

    def test_the_cross_contrast_route_refuses_it(self, tmp_path) -> None:
        files = []
        for i, name in enumerate(("2022_T101.h5", "2022_T201.h5")):
            _write(tmp_path / name, seed=i)
            files.append(tmp_path / name)
        ds = M4RawRepetitionDataset(
            h5_files=files,
            single_contrast=False,
            coil_processing_mode="none",
            target_mode="nullspace_r2r",
        )
        with pytest.raises(ValueError, match="single-contrast denoising target"):
            _ = ds[0]


class TestAlternateInput:
    _REPS = ("01", "02", "03")

    def _loo(self, tmp_path, **kw) -> M4RawRepetitionDataset:
        return _dataset(
            tmp_path,
            reps=self._REPS,
            target_mode="phase_aligned_mean",
            nex_target_exclude_input=True,
            emit_alternate_input=True,
            **kw,
        )

    def test_it_is_repetition_one_in_the_inputs_phase_frame(self, tmp_path) -> None:
        ds = self._loo(tmp_path)
        subject = ds[0]
        paths = [m4raw_mod.Path(p) for p in ds.index[0]["paths"]]
        reps = m4raw_mod._load_reps_or_skip(paths, "alt reference", None)
        expected = m4raw_mod._phase_align(reps[1], reps[0])
        assert torch.allclose(_complex(subject["input_alt"].data), expected, atol=1e-4)
        assert subject["input_alt"].data.shape == subject["input"].data.shape

    def test_it_is_aligned_not_raw(self, tmp_path) -> None:
        """A repetition with a receiver phase offset must arrive rotated back."""
        ds = self._loo(tmp_path)
        subject = ds[0]
        inp, alt = _complex(subject["input"].data), _complex(subject["input_alt"].data)
        dot = (alt * inp.conj()).sum(dim=(-2, -1))
        assert float(dot.angle().abs().max()) < 1e-4

    def test_off_by_default(self, tmp_path) -> None:
        ds = _dataset(tmp_path, reps=self._REPS, target_mode="phase_aligned_mean")
        assert "input_alt" not in ds[0]

    def test_a_split_training_split_publishes_nothing(self, tmp_path) -> None:
        """The knob is shared with the val split; r2r has no second repetition to use."""
        ds = _dataset(tmp_path, reps=self._REPS, target_mode="r2r", emit_alternate_input=True)
        assert "input_alt" not in ds[0]

    def test_refused_without_repetitions(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="use_repetitions=False"):
            M4RawRepetitionDataset(h5_files=[], use_repetitions=False, emit_alternate_input=True)

    def test_refused_on_a_group_with_one_repetition(self, tmp_path) -> None:
        with pytest.raises(ValueError, match=">= 2 repetitions"):
            _dataset(
                tmp_path,
                reps=("01",),
                target_mode="phase_aligned_mean",
                emit_alternate_input=True,
            )

    def test_the_normalization_transform_scales_it_with_the_input(self, tmp_path) -> None:
        from spectramr.data.transforms.normalization import KSpaceNormalizationTransform

        subject = self._loo(tmp_path)[0]
        raw_input, raw_alt = subject["input"].data.clone(), subject["input_alt"].data.clone()
        out = KSpaceNormalizationTransform(scale_domain="image")(subject)
        scale = float(out["kspace_scale"])
        assert torch.allclose(out["input"].data * scale, raw_input, rtol=1e-4)
        assert torch.allclose(out["input_alt"].data * scale, raw_alt, rtol=1e-4)


class TestTheScaleIsMeasuredOnTheMeasurement:
    """Under a split mode ``input`` is ``y + alpha*z``; deployment sees ``y``.

    The split runs in ``__getitem__`` before the transform, so the dataset
    publishes ``y`` under ``MEASURED_INPUT_KEY`` for ``scale_reference="input"``.
    A large ``alpha`` makes the two scales differ by far more than tolerance.
    """

    _ALPHA = 20.0

    @staticmethod
    def _repetition(ds: M4RawRepetitionDataset) -> torch.Tensor:
        paths = [m4raw_mod.Path(p) for p in ds.index[0]["paths"]]
        return m4raw_mod._load_reps_or_skip(paths, "measured reference", None)[0]

    @staticmethod
    def _recording_chain(reference: str | None):
        """A recorder ahead of the k-space normalizer (or alone when *reference* is None)."""
        import torchio as tio

        from spectramr.data.transforms.normalization import (
            MEASURED_INPUT_KEY,
            KSpaceNormalizationTransform,
        )

        seen: dict[str, torch.Tensor | None] = {}

        class _Recorder(tio.Transform):
            def apply_transform(self, subject):
                image = subject.get(MEASURED_INPUT_KEY)
                seen["measured"] = None if image is None else image.data.clone()
                seen["input"] = subject["input"].data.clone()
                return subject

        chain = [_Recorder()]
        if reference is not None:
            chain.append(KSpaceNormalizationTransform(scale_domain="image", scale_reference=reference))
        return tio.Compose(chain), seen

    @pytest.mark.parametrize("mode", ["r2r", "nullspace_r2r"])
    def test_a_consuming_chain_receives_the_repetition_as_acquired(self, tmp_path, mode) -> None:
        from spectramr.data.transforms.normalization import MEASURED_INPUT_KEY

        chain, seen = self._recording_chain("input")
        ds = _dataset(tmp_path, target_mode=mode, transform=chain)
        served = ds[0]
        measured = _complex(seen["measured"])
        assert torch.allclose(measured, self._repetition(ds), atol=1e-5)
        assert not torch.allclose(_complex(seen["input"]), measured, atol=1e-3)
        assert MEASURED_INPUT_KEY not in served

    @pytest.mark.parametrize(
        "reference",
        [None, "target"],
        ids=["normalization-off", "target-reference"],
    )
    @pytest.mark.parametrize("with_chain", [True, False], ids=["recorder", "no-transform"])
    def test_a_split_mode_publishes_nothing_when_nothing_consumes_it(
        self, tmp_path, reference, with_chain
    ) -> None:
        """PLANTED: normalization off (or on the target) must not carry the key."""
        from spectramr.data.transforms.normalization import MEASURED_INPUT_KEY

        chain, seen = self._recording_chain(reference)
        ds = _dataset(tmp_path, target_mode="r2r", transform=chain if with_chain else None)
        served = ds[0]
        assert MEASURED_INPUT_KEY not in served
        if with_chain:
            assert seen["measured"] is None

    def test_an_unsplit_mode_publishes_nothing(self, tmp_path) -> None:
        chain, seen = self._recording_chain("input")
        _dataset(tmp_path, reps=("01", "02"), target_mode="phase_aligned_mean", transform=chain)[0]
        assert seen["measured"] is None

    def test_the_r2r_scale_is_the_measurements_not_the_recorrupted_inputs(self, tmp_path):
        """PLANTED: the scale of ``y + alpha*z`` is the shape this must not take."""
        from spectramr.data.transforms.normalization import (
            MEASURED_INPUT_KEY,
            KSpaceNormalizationTransform,
        )

        transform = KSpaceNormalizationTransform(scale_domain="image", scale_reference="input")
        ds = _dataset(tmp_path, target_mode="r2r", r2r_alpha=self._ALPHA, transform=transform)
        out = ds[0]
        scale = out["kspace_scale"]
        measured = _stacked(self._repetition(ds))
        expected = transform.spec.compute_scale(measured, channel_dim=0)
        recorrupted = transform.spec.compute_scale(out["input"].data * scale, channel_dim=0)
        assert torch.allclose(scale, expected, rtol=1e-4)
        assert float(recorrupted / expected) > 2.0
        assert MEASURED_INPUT_KEY not in out, "read for the scale, never served"

    @pytest.mark.parametrize("mode", ["r2r", "nullspace_r2r"])
    def test_the_measurement_handed_to_the_model_is_the_split_input(self, tmp_path, mode):
        """#2761: the acquired repetition beside the split input determines the target."""
        from spectramr.data.transforms.normalization import KSpaceNormalizationTransform

        transform = KSpaceNormalizationTransform(scale_domain="image", scale_reference="input")
        ds = _dataset(tmp_path, target_mode=mode, transform=transform)
        out = ds[0]
        torch.testing.assert_close(out["measured_kspace"].data, out["input"].data)
        acquired = _stacked(self._repetition(ds)) / out["kspace_scale"]
        assert not torch.allclose(out["measured_kspace"].data, acquired, atol=1e-3)
