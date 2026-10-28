from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from spectramr.config.schemas.metrics import MetricsConfigSchema
from spectramr.infrastructure.training.strategies.disentangled_strategy import (
    DisentangledTrainingStrategy,
)


class MockGenerator(nn.Module):
    def __init__(self, use_vae=True):
        super().__init__()
        self.use_vae = use_vae
        self.enc_c = MagicMock(return_value=torch.randn(1, 4, 32, 32))
        self.enc_s = MagicMock()
        self.gen = MagicMock(return_value=torch.randn(1, 1, 32, 32))
        # validation decodes through decode_tissue(content, condition_style(style, label)).
        self.decode_tissue = MagicMock(return_value=torch.randn(1, 1, 32, 32))
        self.condition_style = MagicMock(side_effect=lambda s, label: s)
        self.reparameterize = MagicMock(return_value=torch.randn(1, 4, 1, 1))

    def forward(self, x):
        return x


def test_validation_step_vae_tuple_handling(tmp_path):
    """Test that validation_step correctly handles tuple output from VAE enc_s."""
    print("\n--- Starting test_validation_step_vae_tuple_handling (Unbound) ---")

    # 1. Mock Strategy Instance
    mock_self = MagicMock()
    mock_self.device = torch.device("cpu")
    mock_self._in_channels = 1

    # `config` is the REAL schema, not a mock leaf. `validation_step` hands
    # `config.metrics.output_dir` to `os.makedirs`, and a MagicMock satisfies
    # `os.PathLike` -- so this one attribute, left auto-mocked, wrote a
    # `MagicMock/mock.config.metrics.output_dir/<id>/val_images/` tree into the
    # repo root every time this test ran (#917), and the assertions below said
    # nothing about it. `MetricsConfigSchema` constructs bare, so there is no
    # shape here to drift from the real one.
    mock_self.config = SimpleNamespace(
        metrics=MetricsConfigSchema(output_dir=str(tmp_path))
    )

    # Mock State and Generator
    mock_self.env.generator = MockGenerator(use_vae=True)
    mock_self.inner_for_call = lambda model, name: model  # single-process (#2629)
    mu = torch.randn(1, 4, 1, 1)
    logvar = torch.randn(1, 4, 1, 1)
    mock_self.env.generator.enc_s.return_value = (mu, logvar)

    # Mock Losses
    mock_self._recon_loss_fn = MagicMock(return_value=torch.tensor(0.5))
    mock_self._mind_ssc_loss_fn = MagicMock(return_value=torch.tensor(0.1))
    # Mock other losses as None or Mocks
    mock_self._hist_loss_fn = None
    mock_self._perceptual_loss_fn = None
    mock_self._lpips_loss_fn = None
    mock_self._dists_loss_fn = None

    # Mock Config
    mock_self.state.config.losses.reconstruction.lambda_l1 = 1.0
    mock_self.state.config.losses.reconstruction.enable_l1 = True
    mock_self.state.config.losses.reconstruction.lambda_mind_ssc = 0.0
    mock_self.state.config.losses.reconstruction.enable_mind_ssc = False
    mock_self.state.config.losses.latent.kl_capacity_target = 10.0
    mock_self.state.config.losses.latent.kl_capacity_warmup_steps = 100
    mock_self.state.config.losses.latent.use_capacity_scheduling = False
    mock_self.state.config.losses.latent.lambda_kl = 0.1

    # Mock Compute Dependencies
    mock_self.validation_metrics_computer.compute.return_value = {"psnr": 20.0}
    mock_self._unpack_batch.side_effect = lambda b: (b["input"], b["target"])

    # 2. Input
    input_batch = torch.randn(1, 1, 32, 32)
    target_batch = torch.randn(1, 1, 32, 32)
    batch = {"input": input_batch, "target": target_batch}

    # 3. Call Method Unbound
    try:
        metrics = DisentangledTrainingStrategy.validation_step(
            mock_self, batch=batch, input_batch=input_batch, target_batch=target_batch
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        pytest.fail(f"validation_step failed: {e}")

    # 4. Assertions
    model = mock_self.env.generator
    assert model.reparameterize.called
    assert model.decode_tissue.called
    args, _ = model.decode_tissue.call_args
    assert torch.is_tensor(args[1]), "Generator received tuple instead of tensor"
    print("--- Passed test_validation_step_vae_tuple_handling ---")


@pytest.mark.skip(
    reason=(
        "UNCOVERED, not covered elsewhere (issue #634). 5-D chunking goes "
        "through TorchIOAdapter.to_batch_format, which rearranges depth-vs-"
        "spatial dims in a way this bare-MagicMock fixture does not track. "
        "The earlier claim that 'the chunking-specific path is exercised in "
        "the integration suite' was false — nothing under tests/integration/ "
        "mentions chunking at all. The 4-D path "
        "(test_validation_step_vae_tuple_handling above) covers the VAE-tuple "
        "branch; the 5-D chunked variant needs a real TorchIOAdapter fixture."
    )
)
def test_validation_step_chunking_vae_tuple_handling():
    """Test chunking path VAE tuple handling."""
    print(
        "\n--- Starting test_validation_step_chunking_vae_tuple_handling (Unbound) ---"
    )

    # 1. Mock Strategy
    mock_self = MagicMock()
    mock_self.device = torch.device("cpu")
    mock_self._in_channels = 1

    # Mock Model
    model = MockGenerator(use_vae=True)
    model.use_vae = True  # Ensure attribute exists for the strategy

    # The 5D chunking path uses chunk_size=1 (one slice at a time) so the
    # encoder/decoder return shapes must follow the actual chunk size, not
    # a fixed 4×.  Use side_effect to compute shapes from the actual call args.
    def _enc_c_side_effect(x_in):
        b = x_in.shape[0]
        return torch.randn(b, 4, 32, 32)

    def _enc_s_side_effect(x_in):
        b = x_in.shape[0]
        return (torch.randn(b, 4, 1, 1), torch.randn(b, 4, 1, 1))

    def _reparameterize_side_effect(mu, logvar):
        return torch.randn_like(mu)

    def _gen_side_effect(c, s):
        b = c.shape[0]
        return torch.randn(b, 1, 32, 32)

    model.enc_c = MagicMock(side_effect=_enc_c_side_effect)
    model.enc_s = MagicMock(side_effect=_enc_s_side_effect)
    model.reparameterize = MagicMock(side_effect=_reparameterize_side_effect)
    model.gen = MagicMock(side_effect=_gen_side_effect)
    mock_self.env.generator = model

    # Mock Losses
    mock_self._recon_loss_fn = MagicMock(return_value=torch.tensor(0.5))

    # Mock Config
    mock_self.state.config.losses.reconstruction.lambda_l1 = 1.0
    mock_self.state.config.losses.reconstruction.enable_l1 = True

    # Mock Compute
    mock_self.validation_metrics_computer.compute.return_value = {"psnr": 20.0}

    # 2. Input (5D): (B=2, C=1, D=16, H=32, W=32) -> 32 slices total
    input_batch = torch.randn(2, 1, 16, 32, 32)
    # Pass 5D target with SAME shape; strategy will flatten it to (32, 1, 32, 32)
    target_batch = input_batch.clone()

    # 3. Call Method
    try:
        metrics = DisentangledTrainingStrategy.validation_step(
            mock_self, batch={}, input_batch=input_batch, target_batch=target_batch
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        pytest.fail(f"Chunking validation failure: {e}")

    # 4. Assertions
    assert model.reparameterize.called
    assert model.gen.called
    args, _ = model.gen.call_args
    assert torch.is_tensor(args[1])
    print("--- Passed test_validation_step_chunking_vae_tuple_handling ---")
