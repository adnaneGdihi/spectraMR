"""Smoke tests for HSSC and DTN2S strategy registration + Hermitian/DC ops.

These tests pin three contracts:

1. The strategy factory resolves ``"hssc"`` and ``"dtn2s"`` to the new
   strategy classes without import-time errors.
2. HSSC's Hermitian-symmetry projection actually returns a tensor that
   satisfies ``k = conj(flip(k))`` after one application.
3. HSSC's data-consistency clamp returns observed k-space exactly inside
   the mask region.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.hermitian import is_hermitian
from spectramr.infrastructure.training.strategies.dtn2s_strategy import DTN2SStrategy
from spectramr.infrastructure.training.strategies.hssc_strategy import HSSCStrategy
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory


class TestRegistration:
    def test_strategy_paths_present(self) -> None:
        factory = TrainingStrategyFactory()
        paths = factory.STRATEGY_CLASS_PATHS
        assert "hssc" in paths
        assert "dtn2s" in paths
        assert paths["hssc"].endswith("HSSCStrategy")
        assert paths["dtn2s"].endswith("DTN2SStrategy")

    def test_classes_importable(self) -> None:
        # The classes were already imported at module-load time above
        assert HSSCStrategy is not None
        assert DTN2SStrategy is not None
        assert HSSCStrategy.__bases__[0].__name__ == "BaseTrainingStrategy"
        assert DTN2SStrategy.__bases__[0].__name__ == "BaseTrainingStrategy"


class TestHSSCHelpers:
    """Test HSSC's stateless helper methods directly (no env required)."""

    def test_hermitian_projection_idempotent(self) -> None:
        # Symmetric tensor stays symmetric; one-shot projection of an
        # arbitrary tensor produces a Hermitian-symmetric output.
        from types import SimpleNamespace
        # Construct without __init__ to skip BaseTrainingStrategy DI plumbing
        s = HSSCStrategy.__new__(HSSCStrategy)
        torch.manual_seed(0)
        x = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        x_h = s._hermitian_project(x)
        assert is_hermitian(x_h)
        assert torch.allclose(s._hermitian_project(x_h), x_h, atol=1e-6)

    @pytest.mark.parametrize("size", [(16, 16), (15, 16), (15, 15)])
    def test_hermitian_projection_takes_real_part(self, size: tuple[int, int]) -> None:
        # #2408: with the (N - j) mod N partner the projection is x -> Re x, so a
        # real image passes unchanged (the flip pairing zeroed r1 + r2 = N/2).
        s = HSSCStrategy.__new__(HSSCStrategy)
        torch.manual_seed(2)
        x = torch.complex(torch.randn(1, 1, *size), torch.randn(1, 1, *size))
        x_proj = ifft2c(s._hermitian_project(fft2c(x)))
        assert torch.allclose(x_proj, x.real.to(x.dtype), atol=1e-5)

    def test_data_consistency_inside_mask_exact(self) -> None:
        s = HSSCStrategy.__new__(HSSCStrategy)
        torch.manual_seed(1)
        k_obs = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        k_hat = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        mask[..., 3:5, :] = True   # central two PE lines fixed
        out = s._data_consistency(k_hat, k_obs, mask)
        # Inside mask: must equal k_obs
        assert torch.allclose(out[..., 3:5, :], k_obs[..., 3:5, :])
        # Outside mask: must equal k_hat
        assert torch.allclose(out[..., :3, :], k_hat[..., :3, :])
        assert torch.allclose(out[..., 5:, :], k_hat[..., 5:, :])


class _StubLossOutput:
    def __init__(self, total: torch.Tensor) -> None:
        self.total = total
        self.components: dict[str, torch.Tensor] = {}


class _StubLossComputer:
    """Records the ``iteration`` it was called with; returns a grad-scalar."""

    def __init__(self) -> None:
        self.last_iteration: int | None = None

    def compute(self, *, pred, target, epoch, iteration, losses_dict):  # noqa: ANN001
        self.last_iteration = iteration
        return _StubLossOutput(pred.abs().mean())


class _IdentityGen:
    """Returns its input unchanged (k-space in == k-space out)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestHSSCMaskContract:
    """The sampling mask must be read from the batch, not a top-level kwarg the
    training loop never sets (pre-fix: ``kwargs.get('mask')`` -> always None ->
    HSSC raised at step 1, so ``task_2_1_recon_hssc`` could not run)."""

    def _strategy(self):
        s = HSSCStrategy.__new__(HSSCStrategy)
        s.env = SimpleNamespace(generator=_IdentityGen(), losses={})
        s.loss_computer = _StubLossComputer()
        return s

    def _kspace(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))

    def test_mask_read_from_batch_runs(self) -> None:
        from spectramr.data.batch_types import TrainingBatch

        s = self._strategy()
        k = self._kspace()
        mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        mask[..., 3:5, :] = True
        batch = TrainingBatch(input=k, target=k, mask=mask)
        out = s._compute_losses_impl(k, k, epoch=0, batch=batch)
        assert "g_total_loss" in out

    def test_missing_mask_everywhere_raises(self) -> None:
        from spectramr.data.batch_types import TrainingBatch

        s = self._strategy()
        k = self._kspace()
        batch = TrainingBatch(input=k, target=k, mask=None)
        with pytest.raises(ValueError, match="sampling mask"):
            s._compute_losses_impl(k, k, epoch=0, batch=batch)

    def test_iteration_threaded_from_loop_state(self) -> None:
        """The loss computer must see the real loop iteration, not a frozen 0
        from ``kwargs.get('step', 0)`` (the loop passes ``iteration``)."""
        from spectramr.data.batch_types import TrainingBatch

        s = self._strategy()
        s.loop_state = SimpleNamespace(iteration=42)
        k = self._kspace()
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        batch = TrainingBatch(input=k, target=k, mask=mask)
        s._compute_losses_impl(k, k, epoch=0, batch=batch)
        assert s.loss_computer.last_iteration == 42


class TestDTN2SHelpers:
    """DTN2S forward without env: build mask + check shape semantics."""

    def test_resolve_traversals_shape(self) -> None:
        # Skip __init__ (full DI needed); seed cache attrs manually
        s = DTN2SStrategy.__new__(DTN2SStrategy)
        s._phi_a = None
        s._phi_b = None
        s._mask = None
        x = torch.randn(1, 1, 16, 16)
        phi_a, phi_b = s._resolve_traversals(x)
        assert phi_a.shape == (16, 16)
        assert phi_b.shape == (16, 16)
        # Calling again with same shape returns the cached pair (identity)
        phi_a2, phi_b2 = s._resolve_traversals(x)
        assert phi_a is phi_a2 and phi_b is phi_b2
