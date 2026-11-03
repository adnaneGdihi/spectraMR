r"""Unit tests for the LCAH and DL-BAE training objectives.

Targets the module-level loss functions of
``spectramr.infrastructure.training.strategies.hypernetwork_strategy`` (M3) and
``spectramr.infrastructure.training.strategies.dispersion_bloch_ae_strategy`` (M4).

Both strategies keep their science in a module-level function so it is testable
without constructing a trainer -- the same seam as ``bloch_field`` and
``operator_id_bch``. These tests exercise that seam directly.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.dispersion_bloch_ae_strategy import (
    compute_dispersion_bloch_ae_loss,
)
from spectramr.infrastructure.training.strategies.hypernetwork_strategy import (
    compute_lcah_loss,
)
from spectramr.models.encoders.lcah_encoder import LCAHEncoder
from spectramr.models.losses.dispersion_monotonicity_loss import DispersionMonotonicity
from spectramr.models.losses.image.multifield_data_consistency import (
    MultiFieldDataConsistency,
)
from spectramr.models.physics_ae.disp_bloch_ae import DispersionBlochAutoencoder

pytestmark = pytest.mark.unit

FIVE_FIELDS = (0.055, 0.3, 1.5, 3.0, 7.0)


class TestComputeLCAHLoss:
    @staticmethod
    def _model() -> LCAHEncoder:
        torch.manual_seed(0)
        return LCAHEncoder(in_channels=1, acq_dim=5, hidden_channels=8, out_channels=1)

    @staticmethod
    def _batch(n: int = 2) -> dict[str, torch.Tensor]:
        return {
            "input": torch.rand(n, 1, 8, 8),
            "target": torch.rand(n, 1, 8, 8),
            "acquisition": torch.rand(n, 5),
        }

    def test_returns_total_and_grad_carrying_pair(self) -> None:
        out = compute_lcah_loss(self._model(), self._batch())
        assert "loss_total" in out and out["loss_total"].requires_grad
        assert out["prediction"].shape == (2, 1, 8, 8)
        assert "target_image" in out

    def test_missing_acquisition_vector_raises(self) -> None:
        """A hypernetwork with no conditioning vector is meaningless -- fail loudly."""
        batch = self._batch()
        del batch["acquisition"]
        with pytest.raises(KeyError, match="acquisition"):
            compute_lcah_loss(self._model(), batch)

    def test_honours_a_custom_acquisition_key(self) -> None:
        batch = self._batch()
        batch["acq_vec"] = batch.pop("acquisition")
        out = compute_lcah_loss(self._model(), batch, acquisition_key="acq_vec")
        assert torch.isfinite(out["loss_total"])

    def test_broadcasts_a_shared_acquisition_vector(self) -> None:
        """One vector for the whole batch is a legitimate single-protocol arm."""
        batch = self._batch(3)
        batch["acquisition"] = torch.rand(5)
        out = compute_lcah_loss(self._model(), batch)
        assert out["prediction"].shape[0] == 3

    def test_lipschitz_penalty_is_added_only_when_configured(self) -> None:
        model, batch = self._model(), self._batch()
        plain = compute_lcah_loss(model, batch)
        assert "loss_lipschitz_budget" not in plain
        penalised = compute_lcah_loss(
            model, batch, lipschitz_weight=1.0, lipschitz_target=1e-6
        )
        assert "loss_lipschitz_budget" in penalised
        assert penalised["loss_total"].item() > plain["loss_total"].item()

    def test_zero_loss_when_prediction_matches_target(self) -> None:
        # eval(): in train mode the spectral-norm power iteration updates its
        # u/v vectors on every forward, so the reference pass and the scored
        # pass would use marginally different weights.
        model, batch = self._model().eval(), self._batch()
        with torch.no_grad():
            batch["target"] = model(batch["input"], batch["acquisition"])
        out = compute_lcah_loss(model, batch)
        assert out["loss_l1"].item() == pytest.approx(0.0, abs=1e-6)


class TestComputeDispersionBlochAELoss:
    @staticmethod
    def _model() -> DispersionBlochAutoencoder:
        torch.manual_seed(0)
        return DispersionBlochAutoencoder(
            fields_present=FIVE_FIELDS,
            n_pools=1,
            hidden_channels=8,
            depth=2,
            repetition_times_s=(0.5,) * len(FIVE_FIELDS),
            echo_times_s=(0.015,) * len(FIVE_FIELDS),
        )

    def _call(self, batch, **kw):
        return compute_dispersion_bloch_ae_loss(
            self._model(),
            batch,
            data_consistency=MultiFieldDataConsistency(),
            monotonicity=DispersionMonotonicity(),
            **kw,
        )

    def test_returns_both_terms_and_a_differentiable_total(self) -> None:
        out = self._call({"input": torch.rand(2, 5, 8, 8)})
        assert out["loss_total"].requires_grad
        assert "loss_multifield_data_consistency" in out
        assert "loss_dispersion_monotonicity" in out
        out["loss_total"].backward()

    def test_autoencodes_input_when_no_target_supplied(self) -> None:
        """DL-BAE is an autoencoder: the observed stack IS the reference."""
        x = torch.rand(1, 5, 6, 6)
        out = self._call({"input": x})
        assert torch.equal(out["target_image"], x)

    def test_uses_an_explicit_target_when_present(self) -> None:
        x, y = torch.rand(1, 5, 6, 6), torch.rand(1, 5, 6, 6)
        out = self._call({"input": x, "target": y})
        assert torch.equal(out["target_image"], y)

    def test_missing_input_raises(self) -> None:
        with pytest.raises(KeyError, match="multi-field stack"):
            self._call({"target": torch.rand(1, 5, 6, 6)})

    def test_weights_scale_the_respective_terms(self) -> None:
        batch = {"input": torch.rand(1, 5, 6, 6)}
        base = self._call(batch, data_consistency_weight=1.0, monotonicity_weight=0.0)
        scaled = self._call(batch, data_consistency_weight=2.0, monotonicity_weight=0.0)
        assert scaled["loss_total"].item() == pytest.approx(
            2.0 * base["loss_total"].item(), rel=1e-5
        )

    def test_monotonicity_term_sees_the_model_field_grid(self) -> None:
        """The hinge must be evaluated on sorted fields, not raw channel order."""
        out = self._call({"input": torch.rand(1, 5, 6, 6)})
        assert torch.isfinite(out["loss_dispersion_monotonicity"])
        assert out["loss_dispersion_monotonicity"].item() >= 0.0
