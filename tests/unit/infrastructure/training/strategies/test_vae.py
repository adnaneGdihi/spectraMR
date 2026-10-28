"""VAETrainingStrategy autoencoder-target guard (2026-07 ldm_two_stage triage).

The stage-1 VAE reconstruction target MUST equal the input (an autoencoder).
The former ``target_batch = input_batch`` fallback silently rescued a
shape-mismatched target — which is exactly how a ``hf_to_ulf`` *translation*
config masqueraded as an HF autoencoder and corrupted the frozen stage-2 latent.
The guard now RAISES on a missing / empty / shape-mismatched target. These tests
pin that a matching-shape target proceeds and a mismatched / None target raises.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from spectramr.core.module_utils import WRAPPER_ATTRS
from spectramr.infrastructure.training.strategies.vae import VAETrainingStrategy


class _SentinelError(Exception):
    """Raised by the mocked loss computer to prove control reached past the guard."""


def _mock_self(recon: torch.Tensor):
    """A MagicMock strategy whose generator returns (recon, mu, logvar) and whose
    loss computer raises _SentinelError — so a passing guard surfaces as _SentinelError and
    a failing guard surfaces as ValueError, cleanly distinguishing the two."""
    mock = MagicMock()
    mu = torch.zeros(recon.shape[0], 4, 1, 1)
    logvar = torch.zeros(recon.shape[0], 4, 1, 1)
    mock.env.generator = MagicMock(return_value=(recon, mu, logvar))
    mock.precision_manager.prepare_latent_for_kl_computation = MagicMock(return_value=(mu, logvar))
    mock.env.losses = {}
    mock.loss_computer.compute = MagicMock(side_effect=_SentinelError)
    return mock


def test_compute_losses_matching_target_passes_guard() -> None:
    inp = torch.randn(1, 1, 32, 32)
    recon = torch.randn(1, 1, 32, 32)  # same shape as target below
    mock = _mock_self(recon)
    # target matches recon → guard must NOT raise ValueError; control reaches the
    # loss computer (our sentinel), proving the autoencoder objective was accepted.
    with pytest.raises(_SentinelError):
        VAETrainingStrategy._compute_losses_impl(
            mock, inp, torch.randn(1, 1, 32, 32), epoch=0, iteration=0
        )


def test_compute_losses_mismatched_target_raises() -> None:
    inp = torch.randn(1, 1, 32, 32)
    recon = torch.randn(1, 1, 32, 32)
    mock = _mock_self(recon)
    with pytest.raises(ValueError, match="target ≡ input"):
        VAETrainingStrategy._compute_losses_impl(
            mock, inp, torch.randn(1, 1, 16, 16), epoch=0, iteration=0
        )


def test_compute_losses_none_target_raises() -> None:
    inp = torch.randn(1, 1, 32, 32)
    recon = torch.randn(1, 1, 32, 32)
    mock = _mock_self(recon)
    with pytest.raises(ValueError, match="missing or shape-mismatched"):
        VAETrainingStrategy._compute_losses_impl(mock, inp, None, epoch=0, iteration=0)


class TestBareTensorPosteriorRecovery:
    """A VAE that returns a BARE TENSOR but caches its posterior must still get a KL.

    ``slat_vae_slab_to_volume`` returns a plain tensor from ``forward`` (its
    structured return is behind an opt-in ``return_structured=True`` that this path
    never passes) and exposes the posterior via ``last_aux()``. The strategy used to
    fall straight through to dummy ZERO mu/logvar, making KL identically 0: the beta
    schedule was routed correctly and then multiplied a constant zero, so the "VAE"
    trained as a plain autoencoder with an unregularised latent (pitfalls #9/#16).
    """

    @staticmethod
    def _loss_output():
        """A real-enough LossOutput: the strategy reads .total.device downstream."""
        return SimpleNamespace(
            total=torch.tensor(1.0, requires_grad=True), components={}, metrics={}
        )

    @classmethod
    def _mock_with_last_aux(cls, recon, mu, logvar):
        mock = MagicMock()
        mock.env.generator = MagicMock(return_value=recon)  # BARE tensor
        # A MagicMock fabricates every wrapper attribute; pin them to None so
        # unwrap_model treats this as a bare module, as a real generator is.
        for wrapper_attr in WRAPPER_ATTRS:
            setattr(mock.env.generator, wrapper_attr, None)
        mock.env.generator.last_aux = MagicMock(
            return_value={"mu": mu, "logvar": logvar, "s_coarse": None}
        )
        # Identity precision manager so we can assert on the values that arrive.
        mock.precision_manager.prepare_latent_for_kl_computation = MagicMock(
            side_effect=lambda m, lv: (m, lv)
        )
        mock.env.losses = {}
        mock.loss_computer.compute = MagicMock(return_value=cls._loss_output())
        return mock

    def test_posterior_is_pulled_from_last_aux(self) -> None:
        inp = torch.randn(1, 1, 32, 32)
        recon = torch.randn(1, 1, 32, 32)
        mu = torch.full((1, 8, 4, 4), 0.5)
        logvar = torch.full((1, 8, 4, 4), -0.25)

        mock = self._mock_with_last_aux(recon, mu, logvar)
        VAETrainingStrategy._compute_losses_impl(mock, inp, recon.clone(), epoch=0, iteration=0)

        posterior = mock.loss_computer.compute.call_args.kwargs["posterior"]
        got_mu, got_logvar = posterior

        # The REAL posterior arrived — not the 1-element zero dummy.
        assert torch.equal(got_mu, mu)
        assert torch.equal(got_logvar, logvar)
        # The pre-fix bug in one assertion: a zero posterior yields KL == 0 exactly.
        assert got_mu.abs().sum() > 0

    def test_plain_non_vae_model_still_gets_a_zero_posterior(self) -> None:
        """A model with no ``last_aux`` (e.g. a bare UNet) keeps the old behaviour:
        reconstruction-only with a zero KL contribution, not a crash."""
        inp = torch.randn(1, 1, 32, 32)
        recon = torch.randn(1, 1, 32, 32)

        mock = MagicMock()
        mock.env.generator = MagicMock(return_value=recon)
        # A MagicMock auto-creates attributes, so last_aux must be explicitly absent.
        del mock.env.generator.last_aux
        mock.precision_manager.prepare_latent_for_kl_computation = MagicMock(
            side_effect=lambda m, lv: (m, lv)
        )
        mock.env.losses = {}
        mock.loss_computer.compute = MagicMock(return_value=self._loss_output())

        VAETrainingStrategy._compute_losses_impl(mock, inp, recon.clone(), epoch=0, iteration=0)

        got_mu, got_logvar = mock.loss_computer.compute.call_args.kwargs["posterior"]
        assert torch.count_nonzero(got_mu) == 0
        assert torch.count_nonzero(got_logvar) == 0


# --------------------------------------------------------------------------- #
# VQVAETrainingStrategy: the train-metric iteration seam (#1937)
#
# ``VQVAETrainingStrategy._compute_losses_impl`` fed the metric throttle
# ``getattr(self.env, "step", 0)``. ``TrainingEnvironment`` is ``frozen=True`` and
# declares no ``step``, so the value was a constant 0 -- and ``0 % interval == 0`` for
# every interval, so the host-syncing SSIM/PSNR/MAE were recomputed on EVERY step.
#
# The guard that existed to forbid exactly this line
# (``test_train_metric_step_wiring.py``) named this module and passed green: its walk
# returned on the first class ``inspect.getmembers`` yields, alphabetically
# ``VAETrainingStrategy`` (lines 118-301), and never reached ``VQVAETrainingStrategy``
# (lines 525-620) where the read lived. That walk is fixed in the same change; these
# tests observe the value arrive (non-negotiable 16) rather than reading the source.
# --------------------------------------------------------------------------- #

from spectramr.infrastructure.training.loop_state import LoopState  # noqa: E402
from spectramr.infrastructure.training.strategies.vae import VQVAETrainingStrategy  # noqa: E402

#: Not 0, and not the schema-default train_metric_interval.
_LIVE_ITERATION = 1337


def _vqvae_with_spy(iteration: int):
    """A VQ-VAE stitched with only what the hook's metric branch touches."""
    s = VQVAETrainingStrategy.__new__(VQVAETrainingStrategy)
    # A plain-tensor generator output keeps vq_loss/perplexity None, so the
    # precision manager is not on this path.
    s.state = SimpleNamespace(generator=lambda x: x, model=None)
    s.env = SimpleNamespace(losses={})
    s.config = SimpleNamespace()
    s._loss_dict_reuse = {}
    s.loop_state = LoopState(iteration=iteration, epoch=1)
    s.loss_computer = SimpleNamespace(
        compute=lambda **kw: SimpleNamespace(
            total=torch.zeros((), requires_grad=True), components={}
        )
    )
    seen: list[int] = []

    def _spy(pred, target, config, current_step):
        seen.append(current_step)
        return {}

    s._compute_training_metrics = _spy
    return s, seen


@pytest.mark.unit
def test_vqvae_train_metrics_receive_the_live_iteration_not_a_frozen_zero() -> None:
    strategy, seen = _vqvae_with_spy(_LIVE_ITERATION)
    x = torch.rand(1, 1, 8, 8)
    strategy._compute_losses_impl(input_batch=x, target_batch=x, epoch=1)
    assert seen == [_LIVE_ITERATION], (
        "the train-metric throttle must be fed the live loop iteration; frozen at 0 "
        "it recomputed SSIM/PSNR/MAE on every step (pitfall #16, #1937)"
    )


@pytest.mark.unit
def test_vqvae_iteration_advances_with_the_loop() -> None:
    strategy, seen = _vqvae_with_spy(0)
    x = torch.rand(1, 1, 8, 8)
    strategy._compute_losses_impl(input_batch=x, target_batch=x, epoch=0)
    strategy.loop_state.iteration = _LIVE_ITERATION
    strategy._compute_losses_impl(input_batch=x, target_batch=x, epoch=0)
    assert seen == [0, _LIVE_ITERATION]


# ---------------------------------------------------------------------------
# pillar_05: the orthogonality term and the deterministic-AE comparator.
# ---------------------------------------------------------------------------

_P05 = "experiments/inprogress/pillars/exp_pillar_05_{}.yaml"


def _p05_strategy(arm: str, *, deterministic: bool, losses: dict) -> VAETrainingStrategy:
    from spectramr.config.settings import TrainingSettings
    from spectramr.models.vae.vae import VAE
    from tests.utils.mock_environment import create_mock_training_env

    torch.manual_seed(0)
    model = VAE(in_channels=1, latent_dim=8, hidden_dims=(8, 8, 8, 8), deterministic=deterministic)
    env = create_mock_training_env(
        config=TrainingSettings.from_yaml(_P05.format(arm)), device="cpu", generator=model
    )
    env.losses = losses
    return VAETrainingStrategy(
        env=env, logging_service=MagicMock(), metrics_service=MagicMock(), checkpoint_service=MagicMock()
    )


def _ortho_losses() -> dict:
    from spectramr.models.losses.latent_orthogonality_loss import LatentOrthogonalityLoss

    return {"mse": torch.nn.MSELoss(), "latent_orthogonality": LatentOrthogonalityLoss(split=4)}


def test_the_orthogonality_term_reaches_the_vae_objective() -> None:
    st = _p05_strategy("complex_vae_disentangle", deterministic=False, losses=_ortho_losses())
    st._compute_training_metrics = lambda **_: {}  # ms_ssim needs > 160 px
    x = torch.rand(2, 1, 32, 32)
    out = st._compute_losses_impl(x, x.clone(), 0, iteration=1)
    assert out["latent_orthogonality"] > 0
    # The declared list weight, not 0: the term moves the objective.
    assert st.loss_computer._get_loss_weight("latent_orthogonality", 0, iteration=1) == 0.1
    out["g_total_loss"].backward()


def test_validation_reports_the_orthogonality_residual() -> None:
    st = _p05_strategy("complex_vae_disentangle", deterministic=False, losses=_ortho_losses())
    st._get_validation_metrics_computer = MagicMock(side_effect=RuntimeError("not graded here"))
    x = torch.rand(2, 1, 32, 32)
    metrics = st.validation_step(None, input_batch=x, target_batch=x.clone())
    assert metrics["latent_orthogonality"] >= 0.0


def test_a_deterministic_vae_with_a_kl_weight_is_refused() -> None:
    """Planted: the treatment's KL schedule (0 -> 1) on a z = mu model."""
    with pytest.raises(ValueError, match="deterministic: true decodes z = mu"):
        _p05_strategy("complex_vae_disentangle", deterministic=True, losses={})


def test_the_deterministic_ae_comparator_is_accepted() -> None:
    st = _p05_strategy("control_deterministic_ae", deterministic=True, losses={})
    assert st.loss_computer._get_loss_weight("kl", 0, iteration=10**6) == 0.0


def test_the_orthogonality_term_refuses_a_batch_of_one() -> None:
    st = object.__new__(VAETrainingStrategy)
    st.env = SimpleNamespace(generator=torch.nn.Identity(), losses=_ortho_losses())
    st.config = SimpleNamespace(data=SimpleNamespace(loader=SimpleNamespace(batch_size=1)))
    with pytest.raises(ValueError, match="no second sample"):
        st._refuse_inconsistent_latent_knobs()
