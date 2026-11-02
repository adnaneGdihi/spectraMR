"""prompt_lora_hermitian_fno: registration, identity at init, prompt-driven LoRA update."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.models.generators.prompt_lora_fno import PROMPT_KWARG, PromptLoRAHermitianFNO
from spectramr.models.registry import get_model_class, is_strict_kwargs


def _model(**overrides) -> PromptLoRAHermitianFNO:
    kwargs = {
        "in_channels": 2,
        "out_channels": 2,
        "width": 8,
        "modes": 4,
        "depth": 2,
        "lora_rank": 2,
        "prompt_embed_dim": 8,
    }
    kwargs.update(overrides)
    torch.manual_seed(0)
    return PromptLoRAHermitianFNO(**kwargs)


def _prompts() -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.tensor([[2750.0, 27.0, 90.0, 2.89], [2750.0, 27.0, 90.0, 2.89]])
    b = torch.tensor([[3000.0, 33.0, 140.0, 1.49], [3000.0, 33.0, 140.0, 1.49]])
    return a, b


def test_registered_with_strict_kwargs() -> None:
    assert get_model_class("prompt_lora_hermitian_fno") is PromptLoRAHermitianFNO
    assert is_strict_kwargs("prompt_lora_hermitian_fno")


def test_prompt_has_no_effect_at_init_and_does_after_the_update_moves() -> None:
    model = _model()
    x = torch.randn(2, 2, 16, 16)
    a, b = _prompts()
    assert torch.allclose(model(x, **{PROMPT_KWARG: a}), model(x, **{PROMPT_KWARG: b}))
    for block in model.pointwise:
        nn.init.normal_(block.lora_up.weight, std=0.5)
    out_a = model(x, **{PROMPT_KWARG: a})
    assert out_a.shape == (2, 2, 16, 16)
    assert not torch.allclose(out_a, model(x, **{PROMPT_KWARG: b}))


def test_prompt_is_required_and_shape_checked() -> None:
    model = _model()
    x = torch.randn(2, 2, 16, 16)
    with pytest.raises(TypeError):
        model(x)
    with pytest.raises(ValueError, match="acquisition_prompt"):
        model(x, **{PROMPT_KWARG: torch.ones(3, 4)})


def test_rank_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="hermitian_fno control"):
        _model(lora_rank=0)
