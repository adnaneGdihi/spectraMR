"""The n2n arms' validation metrics are finite on data shaped like M4Raw (#2241).

The 2026-09-20 run reported ``val_psnr``/``val_ssim``/``val_zf_psnr`` as NaN on
every arm. The transformed target was correct -- ``ifft_magnitude`` fired and
produced the RSS image -- but ``kspace_scale_domain: image`` pins that image's
99th percentile to 1.0, not its peak, and M4Raw's brightest voxels land at
~2.05. ``resolve_image_data_range`` refuses any target peaking above 2.0, so
PSNR and SSIM went not-applicable. The arms now declare the range the
normalization establishes.

These drive the real ``ModelValidationMixin.validation_step`` ->
``MetricsMixin._apply_metric_transforms`` -> ``ValidationMetricsComputer`` path
with each arm's own loaded config and a spy forward, so the planted shape is the
one the cluster saw: 8 interleaved k-space channels, 4-D and 5-D.
"""

from __future__ import annotations

import math
import types
from pathlib import Path

import pytest
import torch
import yaml

from spectramr.config.settings import TrainingSettings
from spectramr.data.transforms.normalization import kspace_image_domain_scale
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin
from spectramr.infrastructure.training.strategies.mixins.model_validation import (
    ModelValidationMixin,
)

_COHORT = Path(__file__).resolve().parents[3] / "experiments" / "inprogress" / "n2n"
#: Every arm, including the projection arm F and the theta sweep: the batch the
#: host builds carries the maps, the alternate input, the scale and the
#: covariance their scoring paths read.
_ARMS = sorted(_COHORT.glob("n2n_*.yaml"))


class _Host(ModelValidationMixin, MetricsMixin):
    """The two mixins ReconstructionTrainingStrategy validates through, with a spy forward."""

    capabilities = types.SimpleNamespace(emitted_metrics=frozenset())

    def __init__(self, config: TrainingSettings, prediction: torch.Tensor) -> None:
        self.config = config
        self.device = "cpu"
        self.logging_service = None
        self.state = types.SimpleNamespace(config=config)
        self.generator_model = torch.nn.Identity()
        self._prediction = prediction

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        # A real network reads the input it is handed; the 1e-6 term keeps the
        # spy that honest, so the pred_alt guard (#2595) sees input_alt matter.
        return self._prediction + 1e-6 * input_batch.reshape_as(self._prediction)


def _interleave(kspace: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W)`` complex -> ``(B, 2C, H, W)`` as ``[R0, I0, R1, I1, ...]``."""
    out = torch.empty(kspace.shape[0], 2 * kspace.shape[1], *kspace.shape[2:])
    out[:, 0::2] = kspace.real
    out[:, 1::2] = kspace.imag
    return out


def _coil_maps(size: int = 64) -> torch.Tensor:
    """Four smooth unit-norm maps, ``(C, H, W)`` complex, with distinct coil phases."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    centres = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.5), (-1.0, 1.0, 1.0), (1.0, 1.0, 1.5)]
    maps = torch.stack(
        [
            torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) * torch.exp(1j * torch.tensor(p))
            for a, b, p in centres
        ]
    )
    return maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()


def _batch_data(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
    """The validation-batch fields the arms' scoring paths read, in ``target``'s layout.

    The Monte-Carlo estimator needs ``kspace_scale`` (r2r) or complex maps
    (nullspace_r2r) and, on the self-calibrated arm, ``noise_covariance``; the
    coil-map and repetition metrics need the maps and ``input_alt``.
    """
    from spectramr.infrastructure.physics.m4raw_noise import single_repetition_covariance

    batch = target.shape[0]
    maps = _coil_maps(target.shape[2]).unsqueeze(0).repeat(batch, 1, 1, 1)
    alt = prediction + 0.01 * torch.randn_like(prediction)
    if target.ndim == 5:
        maps = maps.unsqueeze(-1)
    return {
        "kspace_scale": torch.ones(batch),
        "sensitivity_complex": maps,
        "input_alt": alt,
        "noise_covariance": single_repetition_covariance().unsqueeze(0).repeat(batch, 1, 1),
    }


def _m4raw_like_batch(peak_over_p99: float = 2.05, size: int = 64) -> torch.Tensor:
    """8-channel k-space whose RSS image has p99 = 1.0 and a peak ~``peak_over_p99``.

    Scaled by the transform layer's own ``kspace_image_domain_scale``, which is
    what ``KSpaceNormalizationTransform`` applies under ``kspace_scale_domain:
    image``.
    """
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    anatomy = ((xx**2 + yy**2) < 0.6).float()
    maps = _coil_maps(size)
    # A small bright feature: under 1% of the voxels, so it moves the peak and
    # leaves the 99th percentile where the normalization puts it.
    anatomy[size // 3 : size // 3 + 3, size // 2 : size // 2 + 3] = peak_over_p99
    kspace = _interleave(fft2c(maps * anatomy).unsqueeze(0).repeat(2, 1, 1, 1))
    return kspace / kspace_image_domain_scale(kspace[0], percentile=0.99, channel_dim=0)


def _rss_peak(kspace8: torch.Tensor) -> float:
    image = ifft2c(torch.complex(kspace8[:, 0::2], kspace8[:, 1::2]))
    return float(image.abs().pow(2).sum(1).sqrt().max())


def _validate(config: TrainingSettings, five_d: bool) -> dict[str, float]:
    torch.manual_seed(0)
    target = _m4raw_like_batch()
    prediction = target + 0.01 * torch.randn_like(target)
    if five_d:
        target, prediction = target.unsqueeze(-1), prediction.unsqueeze(-1)
    return _Host(config, prediction).validation_step(
        prediction, target, batch_data=_batch_data(target, prediction)
    )


def test_the_planted_target_is_the_cluster_shape() -> None:
    """Precondition: p99-normalized RSS peaking just past the resolver's 2x guard."""
    peak = _rss_peak(_m4raw_like_batch())
    assert 2.0 < peak < 2.3, peak


@pytest.mark.parametrize("five_d", [False, True], ids=["4d", "5d"])
@pytest.mark.parametrize("arm", _ARMS, ids=lambda p: p.stem)
def test_every_arm_reports_finite_image_metrics(arm: Path, five_d: bool) -> None:
    metrics = _validate(TrainingSettings.from_yaml(arm), five_d)
    for key in ("val_psnr", "val_ssim", "val_zf_psnr"):
        assert key in metrics, sorted(metrics)
        assert math.isfinite(metrics[key]), (arm.stem, key, metrics)


@pytest.mark.parametrize("five_d", [False, True], ids=["4d", "5d"])
@pytest.mark.parametrize("arm", _ARMS, ids=lambda p: p.stem)
def test_the_scale_contract_changes_no_score_of_a_declared_arm(
    arm: Path, five_d: bool, monkeypatch
) -> None:
    """Every metric the arm computes, with and without the contract, on the production path.

    The declaration wins in the resolver, so the contract can only reach these
    arms as the extra ``scale_contract`` keyword every ``**kwargs`` metric now
    receives; a metric that forwarded it somewhere strict would turn NaN here.
    """
    from spectramr.infrastructure.training.utils import metric_data_range

    config = TrainingSettings.from_yaml(arm)
    assert metric_data_range.scale_contract_from_config(config) is not None
    with_contract = _validate(config, five_d)
    monkeypatch.setattr(metric_data_range, "scale_contract_from_config", lambda config: None)
    without = _validate(TrainingSettings.from_yaml(arm), five_d)
    assert with_contract.keys() == without.keys()
    for key, value in with_contract.items():
        same = value == without[key] or (math.isnan(value) and math.isnan(without[key]))
        assert same, (key, value, without[key])


@pytest.mark.parametrize("arm", _ARMS, ids=lambda p: p.stem)
def test_every_arm_declares_the_normalizations_reference_level(arm: Path) -> None:
    """Pinned per arm: a range that drifts between arms is a second knob."""
    config = TrainingSettings.from_yaml(arm)
    assert config.data.processing.kspace_scale_domain == "image"
    assert config.data.processing.data_range == 1.0


def _arm_a_without_range(tmp_path: Path) -> TrainingSettings:
    """Arm A with ``data.processing.data_range`` removed: the 2026-09-20 config."""
    raw = yaml.safe_load((_COHORT / "n2n_a_r2r_single_ex_m4raw.yaml").read_text())
    raw["data"]["processing"].pop("data_range")
    planted = tmp_path / "arm_a_without_range.yaml"
    planted.write_text(yaml.safe_dump(raw))
    return TrainingSettings.from_yaml(planted)


def test_without_the_declared_range_the_planted_shape_was_nan(tmp_path, monkeypatch) -> None:
    """The violation this file guards, observed rather than assumed.

    With the scale contract withheld, the resolver is the peak test it was on
    2026-09-20, and the undeclared arm must reproduce the NaN, or the finite
    assertions in this file prove nothing.
    """
    from spectramr.infrastructure.training.utils import metric_data_range

    monkeypatch.setattr(metric_data_range, "scale_contract_from_config", lambda config: None)
    metrics = _validate(_arm_a_without_range(tmp_path), five_d=False)
    assert math.isnan(metrics["val_psnr"]) and math.isnan(metrics["val_ssim"]), metrics
    assert math.isfinite(metrics["val_loss"]), "val_loss never used the range"


def test_without_the_declared_range_the_contract_scores_what_the_declaration_does(
    tmp_path: Path,
) -> None:
    """#2241 at its root: the image-domain p99 scale resolves 1.0 on its own.

    The per-arm declaration stays; this is the proof it can go.
    """
    undeclared = _validate(_arm_a_without_range(tmp_path), five_d=False)
    declared = _validate(
        TrainingSettings.from_yaml(_COHORT / "n2n_a_r2r_single_ex_m4raw.yaml"), five_d=False
    )
    for key in ("val_psnr", "val_ssim", "val_zf_psnr"):
        assert math.isfinite(undeclared[key]), (key, undeclared)
        assert undeclared[key] == declared[key], (key, undeclared[key], declared[key])


def test_an_unapplied_scale_stops_validation_by_name(tmp_path: Path) -> None:
    """A target 2400x off the declared scale raises, not NaN, and not the fallback's warning."""
    from spectramr.core.metrics.data_range_contract import DataRangeContractError

    target = _m4raw_like_batch() * 2400.0
    prediction = target * 0.99
    host = _Host(_arm_a_without_range(tmp_path), prediction)
    with pytest.raises(DataRangeContractError):
        host.validation_step(prediction, target, batch_data=_batch_data(target, prediction))


def test_the_declared_range_changes_no_score_the_resolver_accepts(tmp_path: Path) -> None:
    """Below the guard, declared 1.0 and the auto-resolver agree to the bit."""
    raw = yaml.safe_load((_COHORT / "n2n_a_r2r_single_ex_m4raw.yaml").read_text())
    raw["data"]["processing"].pop("data_range")
    planted = tmp_path / "arm_a_without_range.yaml"
    planted.write_text(yaml.safe_dump(raw))
    target = _m4raw_like_batch(peak_over_p99=1.5)
    prediction = target + 0.01 * torch.randn_like(target)
    batch_data = _batch_data(target, prediction)
    auto = _Host(TrainingSettings.from_yaml(planted), prediction).validation_step(
        prediction, target, batch_data=batch_data
    )
    declared = _Host(
        TrainingSettings.from_yaml(_COHORT / "n2n_a_r2r_single_ex_m4raw.yaml"), prediction
    ).validation_step(prediction, target, batch_data=batch_data)
    assert math.isfinite(auto["val_psnr"])
    for key in ("val_psnr", "val_ssim", "val_zf_psnr"):
        assert declared[key] == auto[key], (key, declared[key], auto[key])


class TestTheFallbackScoresTheTransformedDomain:
    """#2603: when the computer fails, the fallback ``val_psnr`` of a k-space arm.

    The raw tensors are 8-channel interleaved k-space. Scoring them as an image
    gave a finite number in the wrong domain under the arm's declared range, so
    the fallback must score the ``ifft_magnitude`` views or report NaN.
    """

    def _host(self, *, fail_transform: bool) -> tuple[_Host, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        target = _m4raw_like_batch()
        prediction = target + 0.01 * torch.randn_like(target)
        host = _Host(TrainingSettings.from_yaml(_COHORT / "n2n_b_rep_pair_m4raw.yaml"), prediction)
        computer = host.validation_metrics_computer
        context_pass = computer.compute

        def _fail(*args, **kwargs):
            # The coil-space certificate runs first and outside the fallback's
            # try; only the full image-metric pass fails.
            if "context" in kwargs:
                return context_pass(*args, **kwargs)
            raise RuntimeError("planted: the metrics computer failed")

        computer.compute = _fail
        if fail_transform:
            def _no_views(*args, **kwargs):
                raise RuntimeError("planted: the metric transform failed")

            host._metric_transform_views = _no_views
        return host, prediction, target

    @staticmethod
    def _raw_kspace_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
        """What the fallback used to report: channel 0 of raw k-space, range 1.0."""
        mse = torch.nn.functional.mse_loss(prediction[:, :1], target[:, :1]).item()
        return 10.0 * math.log10(1.0 / mse)

    def test_it_scores_the_rss_image_the_computer_would_have(self) -> None:
        host, prediction, target = self._host(fail_transform=False)
        val_config = host.config.validation
        views = host._metric_transform_views(prediction, target, val_config)
        mse = torch.nn.functional.mse_loss(views.pred.float(), views.target.float()).item()
        expected = 10.0 * math.log10(1.0 / mse)

        metrics = host.validation_step(prediction, target, batch_data=_batch_data(target, prediction))
        assert metrics["val_psnr"] == pytest.approx(expected, abs=1e-4)
        assert abs(metrics["val_psnr"] - self._raw_kspace_psnr(prediction, target)) > 1.0

    def test_a_failed_transform_reports_nan_with_the_reason(self, caplog) -> None:
        host, prediction, target = self._host(fail_transform=True)
        with caplog.at_level("WARNING"):
            metrics = host.validation_step(
                prediction, target, batch_data=_batch_data(target, prediction)
            )
        assert math.isnan(metrics["val_psnr"]), metrics
        assert "output_type=kspace" in caplog.text, caplog.text
