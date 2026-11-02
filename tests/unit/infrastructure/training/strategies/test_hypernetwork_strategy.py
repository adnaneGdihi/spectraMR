"""LCAH strategy wiring: the acquisition vector reaches train and validation, and the radius is reported (#2193)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.hypernetwork_strategy import (
    AcquisitionHypernetworkStrategy,
    acquisition_vector,
    require_finite_acquisition,
)
from spectramr.models.encoders.lcah_encoder import LCAHEncoder


def _metadata(n: int = 2) -> dict[str, torch.Tensor]:
    return {
        "TE": torch.full((n,), 98.0),
        "TR": torch.full((n,), 7500.0),
        "TI": torch.full((n,), 1655.0),
        "FA": torch.full((n,), 160.0),
        "B0": torch.full((n,), 0.3),
        "contrast_type": ["inversion_recovery"] * n,
    }


def _strategy() -> AcquisitionHypernetworkStrategy:
    torch.manual_seed(0)
    strategy = object.__new__(AcquisitionHypernetworkStrategy)
    strategy.env = SimpleNamespace(
        generator=LCAHEncoder(in_channels=1, acq_dim=5, hidden_channels=8, out_channels=1)
    )
    strategy._lcah_acquisition_key = "acquisition"
    strategy._lcah_lambda_l1 = 1.0
    strategy._lcah_lipschitz_weight = 0.0
    strategy._lcah_lipschitz_target = None
    strategy._lcah_report_radius = True
    strategy._lcah_train_phi = []
    return strategy


class TestAcquisitionVector:
    def test_stacks_the_metadata_dict_in_canonical_order(self) -> None:
        phi = acquisition_vector({"acquisition_metadata": _metadata(3)})
        assert phi.shape == (3, 5)
        assert torch.allclose(phi[0], torch.tensor([98.0, 7500.0, 1655.0, 160.0, 0.3]))

    def test_a_direct_vector_wins(self) -> None:
        direct = torch.rand(2, 5)
        batch = {"acquisition": direct, "acquisition_metadata": _metadata()}
        assert acquisition_vector(batch) is direct

    def test_missing_source_raises(self) -> None:
        with pytest.raises(KeyError, match="acquisition_metadata"):
            acquisition_vector({"input": torch.rand(2, 1, 4, 4)})

    def test_missing_field_raises(self) -> None:
        meta = _metadata()
        del meta["TI"]
        with pytest.raises(KeyError, match="TI"):
            acquisition_vector({"acquisition_metadata": meta})

    def test_nan_vector_is_refused(self) -> None:
        meta = _metadata()
        meta["TI"] = torch.full((2,), float("nan"))
        with pytest.raises(ValueError, match="not finite"):
            require_finite_acquisition(acquisition_vector({"acquisition_metadata": meta}))


class TestTrainingStep:
    def test_trains_on_the_prepared_tensors_not_the_raw_batch(self) -> None:
        strategy = _strategy()
        strategy._apply_builder_image_losses = lambda pred, target, out: None
        prepared = torch.rand(2, 1, 8, 8)
        # A raw input of the wrong width would fail the 1-channel conv if it were used.
        raw = {"input": torch.rand(2, 4, 8, 8), "acquisition_metadata": _metadata()}
        out = strategy._compute_losses_impl(
            input_batch=prepared, target_batch=torch.rand(2, 1, 8, 8), batch=raw
        )
        assert torch.isfinite(out["loss_total"])
        assert strategy._last_prediction.shape == (2, 1, 8, 8)
        assert strategy._lcah_train_phi[0].shape == (2, 5)


class TestValidation:
    def test_forward_is_conditioned_on_the_context_vector(self) -> None:
        strategy = _strategy()
        x = torch.rand(2, 1, 8, 8)
        phi = acquisition_vector({"acquisition_metadata": _metadata()})
        with torch.no_grad():
            a = strategy._validation_forward(x, {"acquisition": phi})
            b = strategy._validation_forward(x, {"acquisition": phi * 2.0})
        assert a.shape == (2, 1, 8, 8)
        assert not torch.allclose(a, b)

    def test_misaligned_vector_rows_are_refused(self) -> None:
        strategy = _strategy()
        with pytest.raises(ValueError, match="refusing to broadcast"):
            strategy._validation_forward(torch.rand(4, 1, 8, 8), {"acquisition": torch.rand(3, 5)})

    def test_radius_is_zero_on_the_support_and_positive_off_it(self) -> None:
        strategy = _strategy()
        phi = acquisition_vector({"acquisition_metadata": _metadata()})
        strategy._lcah_train_phi = [phi, phi.clone()]
        on = strategy._certificate_metrics(phi)
        off = strategy._certificate_metrics(phi + torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0]))
        assert on["val_lcah_certified_radius"] == 0.0
        assert off["val_lcah_certified_radius"] > 0.0
        assert off["val_lcah_lipschitz_product"] > 0.0
        # The duplicated support collapses to the one distinct protocol.
        assert strategy._lcah_train_phi[0].shape == (1, 5)

    def test_no_support_yet_reports_nothing(self) -> None:
        strategy = _strategy()
        phi = acquisition_vector({"acquisition_metadata": _metadata()})
        assert strategy._certificate_metrics(phi) == {}


def test_acquisition_vector_reads_the_image_collates_per_sample_dicts() -> None:
    """The production image collate leaves ``acquisition_metadata`` as one dict per sample."""
    rows = [
        {"TE": 18.4, "TR": 500.0, "TI": 0.0, "FA": 180.0, "B0": 0.3},
        {"TE": 98.0, "TR": 7500.0, "TI": 1655.0, "FA": 160.0, "B0": 0.3},
    ]
    phi = acquisition_vector({"acquisition_metadata": rows})
    assert torch.allclose(phi, torch.tensor([list(r.values()) for r in rows]))
