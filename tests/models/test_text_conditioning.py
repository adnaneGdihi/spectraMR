import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from spectramr.models.generators.latent_diffusion_generator import (
    LatentDiffusionGenerator,
    LatentDiffusionGeneratorConfig,
)


# Mock transformers classes
class MockCLIPTextModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = 768

    def forward(self, input_ids):
        # Return dummy embeddings [B, Seq, Dim]
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        return (torch.randn(batch_size, seq_len, 768),)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class MockCLIPTokenizer:
    def __init__(self):
        self.model_max_length = 77

    def __call__(self, text, padding, max_length, truncation, return_tensors):
        # Return dummy tokens
        batch_size = len(text)
        return MagicMock(input_ids=torch.randint(0, 1000, (batch_size, max_length)))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


@pytest.fixture
def mock_transformers():
    # The generator imports transformers lazily at construction, so a stand-in
    # module in sys.modules is what the `from transformers import ...` resolves.
    fake = types.ModuleType("transformers")
    fake.CLIPTextModel = MockCLIPTextModel
    fake.CLIPTokenizer = MockCLIPTokenizer
    with patch.dict(sys.modules, {"transformers": fake}):
        yield


def test_text_conditioning_initialization(mock_transformers):
    """Verify initialization with text conditioning enabled."""
    config = LatentDiffusionGeneratorConfig(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        text_conditioning_enabled=True,
        text_model_id="mock-clip",
    )

    model = LatentDiffusionGenerator(config=config)

    assert model.tokenizer is not None
    assert model.text_encoder is not None
    assert model.context_dim == 768
    assert model.denoise_net.context_dim == 768


def test_forward_with_prompts(mock_transformers):
    """Verify forward pass with text prompts."""
    config = LatentDiffusionGeneratorConfig(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=32,  # Small for speed
        text_conditioning_enabled=True,
        text_model_id="mock-clip",
    )

    model = LatentDiffusionGenerator(config=config)

    # Create dummy input
    batch_size = 2
    # Input to forward is noisy latents [B, 4, 16, 16] for 64x64 image
    x = torch.randn(batch_size, 4, 16, 16)
    t = torch.randint(0, 1000, (batch_size,))
    prompts = ["brain mri", "tumor segmentation"]

    # Text conditioning context
    tokens = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    embeddings = model.text_encoder(tokens.input_ids)[0]
    context = {"text_context": embeddings}

    # Run forward
    output = model(x, timesteps=t, context=context)

    # Check output shape
    assert output.shape == x.shape


def test_forward_without_prompts_warning(mock_transformers):
    """Verify behavior when prompts are missing despite config."""
    config = LatentDiffusionGeneratorConfig(
        in_channels=1,
        out_channels=1,
        text_conditioning_enabled=True,
        latent_channels=4,
        base_channels=32,
    )
    model = LatentDiffusionGenerator(config=config)
    x = torch.randn(2, 4, 16, 16)
    t = torch.randint(0, 1000, (2,))

    # Should work but without conditioning (text_context=None)
    output = model(x, timesteps=t)  # No prompts
    assert output.shape == x.shape
