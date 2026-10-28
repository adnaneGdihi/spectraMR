"""Which configurations declare the percentile-scale contract, and the corpus it leaves alone (#2241).

The corpus half is the characterization the change rests on: every
``experiments/inprogress`` arm that declares ``data.processing.data_range``
resolves to exactly that value with the contract in play, and the arms whose
declaration is the #2241 workaround resolve the same value without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from spectramr.config.settings import TrainingSettings
from spectramr.core.metrics.data_range_contract import DataRangeContractError
from spectramr.core.metrics.evaluation_metrics import resolve_image_data_range
from spectramr.core.metrics.outcome import MetricNotApplicableError
from spectramr.infrastructure.training.utils.metric_data_range import (
    scale_contract_from_config,
)

_INPROGRESS = Path(__file__).resolve().parents[5] / "experiments" / "inprogress"


def _declared_range(path: Path) -> float | None:
    """``data.processing.data_range`` as the YAML spells it, without loading the arm."""
    raw = yaml.safe_load(path.read_text()) or {}
    processing = ((raw.get("data") or {}).get("processing")) or {}
    value = processing.get("data_range") if isinstance(processing, dict) else None
    return float(value) if isinstance(value, int | float) else None


_DECLARED = sorted(
    (p for p in _INPROGRESS.rglob("*.yaml") if _declared_range(p) is not None),
    key=str,
)


def _p99_image(peak: float = 2.05, scale: float = 1.0) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    image = ((xx**2 + yy**2) < 0.6).float()
    image[20:23, 30:33] = peak
    return (image * scale).expand(2, 1, 64, 64).clone()


def test_the_corpus_census_still_finds_the_declared_arms() -> None:
    """54 on 2026-10-04 (33 at 2.0, 21 at 1.0); fewer means the YAML read went blind."""
    assert len(_DECLARED) >= 54, len(_DECLARED)


@pytest.mark.parametrize("arm", _DECLARED, ids=lambda p: str(p.relative_to(_INPROGRESS)))
def test_a_declared_range_resolves_unchanged(arm: Path) -> None:
    config = TrainingSettings.from_yaml(str(arm))
    declared = _declared_range(arm)
    assert config.data.processing.data_range == declared
    contract = scale_contract_from_config(config)
    for target in (_p99_image(peak=1.5), _p99_image(), _p99_image(scale=2400.0)):
        got = resolve_image_data_range(target, declared, metric_name="psnr", contract=contract)
        assert got == declared

    if contract is None:
        return
    # The workaround arms: the contract alone gives what the declaration gives,
    # on the very batch that made the sign test report NaN.
    assert declared == 1.0
    with pytest.raises(MetricNotApplicableError):
        resolve_image_data_range(_p99_image(), None, metric_name="psnr")
    assert (
        resolve_image_data_range(_p99_image(), None, metric_name="psnr", contract=contract) == 1.0
    )


class TestWhichConfigurationsDeclareIt:
    # Not an r2r arm: those refuse enable_log_scaling at load, one of the cases below.
    _ARM = _INPROGRESS / "self_supervised" / "ei_unet_m4raw_r4.yaml"

    def _with(self, tmp_path: Path, **processing: object) -> TrainingSettings:
        raw = yaml.safe_load(self._ARM.read_text())
        raw["data"]["processing"].pop("data_range")
        # The arm itself now scales on its ACS lines; the cases start from the image quantile.
        raw["data"]["processing"].update(kspace_scale_domain="image", kspace_scale_reference="target")
        raw["data"]["processing"].update(processing)
        path = tmp_path / "arm.yaml"
        path.write_text(yaml.safe_dump(raw))
        return TrainingSettings.from_yaml(str(path))

    def test_the_m4raw_preset_image_scale_declares_it(self, tmp_path: Path) -> None:
        contract = scale_contract_from_config(self._with(tmp_path))
        assert contract is not None and contract.quantile == 0.99
        assert "kspace_scale_domain=image" in contract.describe()

    @pytest.mark.parametrize(
        "processing",
        [
            {"enable_kspace_normalization": False},
            {"kspace_scale_domain": "kspace"},
            {"enable_log_scaling": True},
            {"kspace_percentile": 1.0},
            {"kspace_scale_domain": "acs"},
            {"kspace_scale_domain": "acs_window"},
        ],
        ids=["off", "kspace-domain", "log-compressed", "max-normalized", "acs", "acs-window"],
    )
    def test_these_do_not(self, tmp_path: Path, processing: dict) -> None:
        assert scale_contract_from_config(self._with(tmp_path, **processing)) is None

    def test_a_kspace_domain_scale_keeps_the_sign_test(self, tmp_path: Path) -> None:
        """The scope: a k-space quantile says nothing about the image peak, so N/A stays."""
        contract = scale_contract_from_config(self._with(tmp_path, kspace_scale_domain="kspace"))
        with pytest.raises(MetricNotApplicableError):
            resolve_image_data_range(_p99_image(), None, metric_name="psnr", contract=contract)

    def test_an_unapplied_scale_under_the_declaration_raises(self, tmp_path: Path) -> None:
        contract = scale_contract_from_config(self._with(tmp_path))
        with pytest.raises(DataRangeContractError):
            resolve_image_data_range(
                _p99_image(scale=2400.0), None, metric_name="psnr", contract=contract
            )

    def test_no_data_block_declares_nothing(self) -> None:
        assert scale_contract_from_config(object()) is None
