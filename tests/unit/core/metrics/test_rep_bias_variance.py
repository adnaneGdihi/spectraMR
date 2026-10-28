"""Rep-BV on synthetic repetitions with a known noise level.

The spy model is the identity, ``f(y) = y``, and an ``N``-repetition average, so
every quantity has a closed form: the identity's variance and error are both
``C sigma^2`` per pixel, its NEX-equivalence is 1, and an ``N``-average's is
``N``. The corrected MSE is checked against the TRUE error (the anatomy is known
here), which is the claim it exists to make.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.coil_manifold import as_interleaved
from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.metric_directions import metric_higher_is_better
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.infrastructure.physics.fft_ops import fft2c

_C, _N, _B = 4, 32, 8
_SIGMA = 0.3


def _randn(*shape: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=gen), torch.randn(*shape, generator=gen))


def _maps() -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _N), torch.linspace(-1, 1, _N), indexing="ij")
    centres = [(-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7), (0.7, 0.7)]
    maps = torch.stack(
        [
            torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) * torch.exp(0.5j * (a * xx + b * yy))
            for a, b in centres
        ]
    ).to(torch.complex64)
    maps = maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()
    return maps.unsqueeze(0).expand(_B, -1, -1, -1)


class _Scene:
    """Anatomy ``x = s m`` and i.i.d. repetitions ``y_k = x + n_k``, E|n|^2 = sigma^2."""

    def __init__(self) -> None:
        self.maps = _maps()
        self.x = self.maps * (2.0 + _randn(_B, 1, _N, _N, seed=0))
        # CN(0, sigma^2): each of Re/Im carries sigma^2 / 2.
        self.reps = [
            self.x + _SIGMA / 2**0.5 * _randn(_B, _C, _N, _N, seed=10 + k) for k in range(12)
        ]

    @staticmethod
    def k(images: torch.Tensor) -> torch.Tensor:
        return as_interleaved(fft2c(images))

    def mean(self, reps: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(reps).mean(0)

    def true_mse(self, images: torch.Tensor) -> float:
        return (images - self.x).abs().pow(2).sum(1).mean().item()


@pytest.fixture(scope="module")
def scene() -> _Scene:
    return _Scene()


def _metric(name: str, prediction, target=None, **context):
    return MetricsRegistry.get(name)(prediction, target, context=MetricContext(**context)).item()


class TestRegistration:
    @pytest.mark.parametrize(
        ("name", "needs", "reference", "higher"),
        [
            ("rep_variance", ("pred_alt",), False, False),
            ("reference_noise_corrected_mse", ("coil_maps",), True, False),
            ("nex_equivalent", ("coil_maps", "y_kspace"), True, True),
            ("bias_squared", ("coil_maps", "pred_alt"), True, False),
        ],
    )
    def test_declared_contract(self, name, needs, reference, higher) -> None:
        assert MetricsRegistry.needs(name) == needs
        assert MetricsRegistry.requires_reference(name) is reference
        assert metric_higher_is_better(name) is higher


class TestTheIdentityModel:
    """``f(y) = y``: variance and error are both ``C sigma^2``; NEX-equivalent 1."""

    def test_rep_variance_is_the_noise_variance(self, scene) -> None:
        value = _metric("rep_variance", scene.k(scene.reps[0]), pred_alt=scene.k(scene.reps[1]))
        assert value == pytest.approx(_C * _SIGMA**2, rel=0.05)

    def test_corrected_mse_matches_the_true_error(self, scene) -> None:
        target = scene.mean(scene.reps[2:6])  # a noisy 4-average reference
        value = _metric(
            "reference_noise_corrected_mse",
            scene.k(scene.reps[0]),
            scene.k(target),
            coil_maps=scene.maps,
        )
        assert value == pytest.approx(scene.true_mse(scene.reps[0]), rel=0.05)

    def test_uncorrected_mse_would_be_off_by_the_reference_noise(self, scene) -> None:
        """Planted: the correction is what brings it to the truth, not luck."""
        target = scene.mean(scene.reps[2:6])
        raw = (scene.reps[0] - target).abs().pow(2).sum(1).mean().item()
        assert raw == pytest.approx(scene.true_mse(scene.reps[0]) * 1.25, rel=0.05)

    def test_nex_equivalent_is_one(self, scene) -> None:
        value = _metric(
            "nex_equivalent",
            scene.k(scene.reps[0]),
            scene.k(scene.mean(scene.reps[2:6])),
            coil_maps=scene.maps,
            y_kspace=fft2c(scene.reps[0]),
        )
        assert value == pytest.approx(1.0, rel=0.07)

    def test_a_non_positive_corrected_mse_gives_nan_not_inf(self, scene) -> None:
        # Prediction equal to the noisy reference: MSE 0, so corrected MSE = -noise < 0.
        k = scene.k(scene.reps[0])
        value = _metric(
            "nex_equivalent", k, k, coil_maps=scene.maps, y_kspace=fft2c(scene.reps[0])
        )
        assert value != value

    def test_the_identity_is_unbiased(self, scene) -> None:
        value = _metric(
            "bias_squared",
            scene.k(scene.reps[0]),
            scene.k(scene.mean(scene.reps[2:6])),
            coil_maps=scene.maps,
            pred_alt=scene.k(scene.reps[1]),
        )
        assert abs(value) < 0.1 * _C * _SIGMA**2


class TestAveragingModels:
    @pytest.mark.parametrize("n", [2, 4])
    def test_an_n_average_is_worth_n_repetitions(self, scene, n: int) -> None:
        value = _metric(
            "nex_equivalent",
            scene.k(scene.mean(scene.reps[:n])),
            scene.k(scene.mean(scene.reps[8:12])),
            coil_maps=scene.maps,
            y_kspace=scene.k(scene.reps[0]),
        )
        assert value == pytest.approx(n, rel=0.1)

    def test_a_shrinking_model_shows_its_bias(self, scene) -> None:
        """``f(y) = 0.8 y``: bias^2 = 0.04 ||x||^2 per pixel, variance 0.64 C sigma^2."""
        f0, f1 = (0.8 * scene.reps[0], 0.8 * scene.reps[1])
        target = scene.k(scene.mean(scene.reps[2:6]))
        corrected = _metric(
            "reference_noise_corrected_mse", scene.k(f0), target, coil_maps=scene.maps
        )
        assert corrected == pytest.approx(scene.true_mse(f0), rel=0.05)
        bias = _metric(
            "bias_squared", scene.k(f0), target, coil_maps=scene.maps, pred_alt=scene.k(f1)
        )
        expected = 0.04 * scene.x.abs().pow(2).sum(1).mean().item()
        assert bias == pytest.approx(expected, rel=0.15)


class TestPhase:
    def test_a_reference_receiver_phase_is_not_scored_as_error(self, scene) -> None:
        """The n2n leave-one-out targets carry a global phase no model can infer."""
        target = scene.mean(scene.reps[2:6])
        aligned = _metric(
            "reference_noise_corrected_mse",
            scene.k(scene.reps[0]),
            scene.k(target),
            coil_maps=scene.maps,
        )
        rotated = _metric(
            "reference_noise_corrected_mse",
            scene.k(scene.reps[0]),
            scene.k(target * torch.exp(torch.tensor(2.0j))),
            coil_maps=scene.maps,
        )
        assert rotated == pytest.approx(aligned, rel=1e-3)

    def test_a_second_repetition_phase_is_not_scored_as_variance(self, scene) -> None:
        plain = _metric("rep_variance", scene.k(scene.reps[0]), pred_alt=scene.k(scene.reps[1]))
        rotated = _metric(
            "rep_variance",
            scene.k(scene.reps[0]),
            pred_alt=scene.k(scene.reps[1] * torch.exp(torch.tensor(-1.3j))),
        )
        assert rotated == pytest.approx(plain, rel=1e-3)


class TestItRefuses:
    @pytest.mark.parametrize(
        ("name", "context", "target", "match"),
        [
            ("rep_variance", {}, False, "pred_alt"),
            ("reference_noise_corrected_mse", {}, True, "coil_maps"),
            ("reference_noise_corrected_mse", {"coil_maps": True}, False, "target"),
            ("nex_equivalent", {"coil_maps": True}, True, "y_kspace"),
            ("bias_squared", {"coil_maps": True}, True, "pred_alt"),
        ],
        ids=["variance-no-alt", "mse-no-maps", "mse-no-target", "nex-no-input", "bias-no-alt"],
    )
    def test_a_missing_field(self, scene, name, context, target, match) -> None:
        if context.get("coil_maps"):
            context = {"coil_maps": scene.maps}
        tgt = scene.k(scene.reps[1]) if target else None
        with pytest.raises(ValueError, match=match):
            _metric(name, scene.k(scene.reps[0]), tgt, **context)

    def test_an_image_domain_caller(self, scene) -> None:
        with pytest.raises(ValueError, match="reads multi-coil k-space"):
            _metric(
                "rep_variance",
                scene.k(scene.reps[0]),
                pred_alt=scene.k(scene.reps[1]),
                signal_domain="image",
            )
