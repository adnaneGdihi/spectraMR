import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from spectramr.infrastructure.inference.diffusion_inference_strategy import (
    DiffusionInferenceStrategy,
)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, x, t=None, **kwargs):
        return x


class TestDiffusionInferenceStrategy(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = MockModel()
        self.config = {
            "diffusion": {
                "num_inference_steps": 10,
                "beta_schedule": "linear",
                "beta_start": 0.0001,
                "beta_end": 0.02,
            }
        }

    def test_initialization(self):
        strategy = DiffusionInferenceStrategy(self.model, self.device, self.config)
        self.assertTrue(hasattr(strategy, "betas"))
        self.assertEqual(len(strategy.betas), 10)
        self.assertTrue(hasattr(strategy, "alphas_cumprod"))

    def test_noise_schedule_linear(self):
        strategy = DiffusionInferenceStrategy(self.model, self.device, self.config)
        # Check start and end betas
        self.assertAlmostEqual(strategy.betas[0].item(), 0.0001, places=6)
        self.assertAlmostEqual(strategy.betas[-1].item(), 0.02, places=6)

    def test_run_inference_shape(self):
        strategy = DiffusionInferenceStrategy(self.model, self.device, self.config)
        # Improve mock model to handle the input shape expected by inference
        # Inference creates random noise of shape (batch, channels, h, w)
        # Default in_channels=1, image_size=256
        # We need to mock performance_optimizer.run_optimized_inference to avoid actual model call issues if shapes mismatch

        strategy.performance_optimizer = MagicMock()
        strategy.performance_optimizer.run_optimized_inference.return_value = torch.zeros(
            1, 1, 32, 32
        )

        # Override config for smaller image to speed up
        strategy.diffusion_config["image_size"] = 32

        # Prepare input noise for unconditional inference
        input_noise = torch.randn(1, 1, 32, 32)
        output = strategy.run_inference(input_tensor=input_noise, batch_size=1, unconditional=True)
        self.assertIsInstance(output, torch.Tensor)
        # Output is (1, 1, 32, 32)
        self.assertEqual(output.shape, (1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# _resolve_diffusion_config + eta wiring (2026-06-12 audit, pitfalls #15/#16)
# ---------------------------------------------------------------------------

import inspect  # noqa: E402

from spectramr.infrastructure.inference import diffusion_inference_strategy as _dis  # noqa: E402
from spectramr.infrastructure.inference.diffusion_inference_strategy import (  # noqa: E402
    _resolve_diffusion_config,
)


class TestResolveDiffusionConfig(unittest.TestCase):
    """The strategy must fall back to the training.diffusion SSOT, mapped."""

    def test_top_level_diffusion_preferred(self):
        cfg = {"diffusion": {"num_inference_steps": 50, "beta_schedule": "cosine"}}
        out = _resolve_diffusion_config(cfg)
        self.assertEqual(out["num_inference_steps"], 50)
        self.assertEqual(out["beta_schedule"], "cosine")

    def test_falls_back_to_training_diffusion_with_key_mapping(self):
        cfg = {"training": {"diffusion": {"sampling_steps": 28, "noise_schedule": "cosine"}}}
        out = _resolve_diffusion_config(cfg)
        # sampling_steps → num_inference_steps; noise_schedule → beta_schedule
        self.assertEqual(out["num_inference_steps"], 28)
        self.assertEqual(out["beta_schedule"], "cosine")

    def test_timesteps_used_when_no_sampling_steps(self):
        cfg = {"training": {"diffusion": {"timesteps": 16}}}
        out = _resolve_diffusion_config(cfg)
        self.assertEqual(out["num_inference_steps"], 16)

    def test_empty_when_neither_present(self):
        self.assertEqual(_resolve_diffusion_config({}), {})
        self.assertEqual(_resolve_diffusion_config({"training": {}}), {})

    def test_noise_schedule_enum_is_coerced_to_str(self):
        class _Enum:
            value = "cosine"

        cfg = {"training": {"diffusion": {"noise_schedule": _Enum()}}}
        out = _resolve_diffusion_config(cfg)
        self.assertEqual(out["beta_schedule"], "cosine")


class TestEtaWiredIntoReverseStep(unittest.TestCase):
    """eta must scale the stochastic term (was validated-but-unused)."""

    def _run(self, eta: float, seed: int):
        config = {"diffusion": {"num_inference_steps": 5, "eta": eta, "image_size": 4}}
        strategy = DiffusionInferenceStrategy(MockModel(), torch.device("cpu"), config)
        strategy.performance_optimizer = MagicMock()
        strategy._predict_noise = lambda x_t, t, conditioning, **kw: torch.zeros_like(x_t)
        torch.manual_seed(seed)
        return strategy.run_inference(torch.ones(1, 1, 4, 4), unconditional=True)

    def test_eta_zero_is_deterministic_and_eta_one_is_not(self):
        self.assertTrue(torch.equal(self._run(0.0, 1), self._run(0.0, 2)))
        self.assertFalse(torch.equal(self._run(1.0, 1), self._run(1.0, 2)))


class TestPredictDataConsistencyLedger(unittest.TestCase):
    """The reverse loop's own hard DC is recorded so the base hook skips."""

    def _strategy(self, apply_at_predict: bool):
        from unittest.mock import patch

        from spectramr.infrastructure.inference import predict_data_consistency as pdc
        from spectramr.models.capabilities import ModelCapabilities

        caps = ModelCapabilities(output_domain="kspace")
        self._patches = [
            patch.dict(pdc.MODEL_REGISTRY, {"stub_diff": {"capabilities": caps}}),
            patch.object(
                pdc, "get_model_capabilities", lambda n: caps if n == "stub_diff" else None
            ),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)
        config = {
            "diffusion": {
                "num_inference_steps": 3,
                "beta_schedule": "linear",
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "in_channels": 2,
                "image_size": 8,
            },
            "physics": {
                "data_consistency": {
                    "enabled": True,
                    "method": "hard",
                    "apply_at_predict": apply_at_predict,
                }
            },
            "model": {"model_type": "stub_diff"},
        }
        strategy = DiffusionInferenceStrategy(MockModel(), torch.device("cpu"), config)
        strategy.performance_optimizer = MagicMock()
        strategy._predict_noise = lambda x_t, t, conditioning, **kw: torch.zeros_like(x_t)
        return strategy

    def test_the_loop_notes_its_projection(self):
        strategy = self._strategy(True)
        self.assertIsNotNone(strategy.predict_dc)
        self.assertIsNotNone(strategy.dc_layer)
        conditioning = torch.randn(1, 2, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., ::2] = 1.0
        applied: list[int] = []
        real = strategy.dc_layer

        def _spy(**kwargs):
            applied.append(1)
            return real(**kwargs)

        strategy.dc_layer = _spy
        strategy.predict_dc.begin()
        out = strategy.run_inference(conditioning, conditional=True, mask=mask)
        self.assertEqual(len(applied), 3, "one hard-DC application per reverse step")
        self.assertTrue(strategy.predict_dc.applied_this_call)
        self.assertEqual(
            strategy.predict_dc_provenance()["applied_by"],
            {"DiffusionInferenceStrategy.run_inference": 1},
        )
        projected = strategy._project_onto_measurement(out, mask=mask, measured_kspace=conditioning)
        self.assertIs(projected, out)

    def test_without_a_mask_the_loop_applies_nothing_and_notes_nothing(self):
        strategy = self._strategy(True)
        strategy.predict_dc.begin()
        strategy.run_inference(torch.randn(1, 2, 8, 8), conditional=True)
        self.assertFalse(strategy.predict_dc.applied_this_call)

    def test_off_knob_records_nothing(self):
        strategy = self._strategy(False)
        self.assertIsNone(strategy.predict_dc)
        self.assertEqual(strategy.predict_dc_provenance(), {"apply_at_predict": False})

    def test_hard_dc_arms_can_be_constructed_at_all(self):
        """``self.logger`` was never bound: every ``enabled: true, method: hard``
        arm raised AttributeError in ``__init__`` before this ledger existed, so
        the in-loop DC path was unreachable from ``spectramr infer``."""
        strategy = self._strategy(False)
        self.assertIsNotNone(strategy.dc_layer)


class TestSamplingUsesTheTrainingSchedule(unittest.TestCase):
    """#2305: `infer` rebuilt a `sampling_steps`-long schedule, so a model trained
    on T=1000 was sampled on T=100 and told indices 0..99."""

    def _strategy(self):
        config = {
            "training": {
                "diffusion": {
                    "timesteps": 1000,
                    "sampling_steps": 100,
                    "noise_schedule": "linear",
                    "in_channels": 1,
                    "image_size": 8,
                }
            }
        }
        return DiffusionInferenceStrategy(MockModel(), torch.device("cpu"), config)

    def test_the_schedule_has_the_training_length(self):
        strategy = self._strategy()
        self.assertEqual(strategy.num_train_timesteps, 1000)
        self.assertEqual(len(strategy.alphas_cumprod), 1000)
        self.assertEqual(len(strategy.sampling_timesteps), 100)

    def test_the_model_is_told_training_indices(self):
        strategy = self._strategy()
        seen: list[int] = []
        strategy.performance_optimizer = MagicMock()

        def _predict(x_t, t, conditioning, **kw):
            seen.append(t)
            return torch.zeros_like(x_t)

        strategy._predict_noise = _predict
        strategy.run_inference(torch.randn(1, 1, 8, 8), unconditional=True)
        self.assertEqual(len(seen), 100)
        self.assertEqual((seen[0], seen[-1]), (999, 9))

    def test_the_oracle_noise_recovers_the_clean_sample(self):
        strategy = self._strategy()
        strategy.performance_optimizer = MagicMock()
        torch.manual_seed(0)
        x0 = torch.rand(1, 1, 8, 8) * 0.8 - 0.4
        eps = torch.randn(1, 1, 8, 8)
        abar = strategy.alphas_cumprod[999]
        x_noisiest = abar.sqrt() * x0 + (1 - abar).sqrt() * eps
        strategy._predict_noise = lambda x_t, t, conditioning, **kw: eps
        out = strategy.run_inference(x_noisiest, unconditional=True)
        self.assertLess((out - x0).abs().max().item(), 1e-4)

    def test_more_sampling_steps_than_training_steps_raises(self):
        config = {"training": {"diffusion": {"timesteps": 10, "sampling_steps": 50}}}
        with self.assertRaises(ValueError):
            DiffusionInferenceStrategy(MockModel(), torch.device("cpu"), config)


def test_it_owns_its_kspace_normalisation() -> None:
    """The sampler normalises and de-normalises, so predict must not divide again (#2581)."""
    from spectramr.infrastructure.inference.diffusion_inference_strategy import (
        DiffusionInferenceStrategy,
    )

    assert DiffusionInferenceStrategy.owns_kspace_normalization is True
