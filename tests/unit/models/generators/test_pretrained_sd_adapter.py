"""PretrainedSDAdapterGenerator: from_pretrained backbone, frozen trunk, ControlNet path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

diffusers = pytest.importorskip("diffusers")

from spectramr.models.generators.pretrained_sd_adapter import (  # noqa: E402
    PretrainedSDAdapterGenerator,
    load_sd_unet,
)

ARM = Path("experiments/inprogress/diffusion/experiment_88_stable_diffusion_medical_adapter.yaml")


def _tiny_unet() -> torch.nn.Module:
    torch.manual_seed(0)
    return diffusers.UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        block_out_channels=(8, 16),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=1,
        cross_attention_dim=8,
        attention_head_dim=4,
        norm_num_groups=4,
    )


@pytest.fixture(scope="module")
def sd_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("sd")
    _tiny_unet().save_pretrained(root / "unet")
    return root


def _perturb_zero_convs(model: PretrainedSDAdapterGenerator) -> None:
    # ControlNet's zero-convs output nothing at init; perturb them so the control image shows.
    cn = model.controlnet
    with torch.no_grad():
        for conv in [*cn.controlnet_down_blocks, cn.controlnet_mid_block]:
            conv.weight.normal_(0, 0.1)
        cn.controlnet_cond_embedding.conv_out.weight.normal_(0, 0.1)


def test_arm_builds_through_model_builder_and_controlnet_fires(sd_dir, tmp_path):
    """The arm's own YAML resolves through ModelBuilder and the ControlNet reads the control image."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

    raw = yaml.safe_load(ARM.read_text())
    raw["model"]["model_kwargs"]["base_model_path"] = str(sd_dir)
    raw["training"]["device"] = "cpu"
    cfg_path = tmp_path / "arm.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = TrainingSettings.from_yaml(str(cfg_path))

    model = ModelBuilder(cfg, torch.device("cpu")).build_generator().build()["generator"]
    assert isinstance(model, PretrainedSDAdapterGenerator)
    assert model.controlnet is not None and model.unet.is_gradient_checkpointing

    _perturb_zero_convs(model)
    fired: list[tuple[int, ...]] = []
    model.controlnet.register_forward_hook(lambda m, a, k: fired.append(tuple(a[0].shape)))

    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    # The strategy's own concat seam builds the [x_t, measurement] input the model splits.
    stub = SimpleNamespace(_conditions_on_input=lambda: True)
    x_t, y = torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16)
    t = torch.tensor([10])

    def run(measurement: torch.Tensor) -> torch.Tensor:
        x_in = DiffusionTrainingStrategy._maybe_condition_on_input(
            stub,
            x_t,
            input_batch=measurement,
            noisy_images=x_t,
            is_cold_diffusion=False,
            is_latent_diffusion=False,
        )
        return model.eval()(x_in, t)

    out = run(y)
    assert out.shape == (1, 1, 16, 16)
    assert fired == [(1, 1, 16, 16)]
    assert not torch.allclose(out, run(torch.zeros_like(y)))

    # One checkpointed training step: grads reach the adapters and never the frozen trunk.
    model.train()
    model(torch.cat([x_t, y], dim=1), t).square().mean().backward()
    trunk = [
        p for n, p in model.unet.named_parameters() if not n.startswith(("conv_in", "conv_out"))
    ]
    assert all(p.grad is None for p in trunk)
    assert model.unet.conv_in.weight.grad is not None
    assert any(p.grad is not None for p in model.controlnet.parameters())


def test_backbone_frozen_adapters_train(sd_dir):
    model = PretrainedSDAdapterGenerator(in_channels=2, out_channels=1, base_model_path=str(sd_dir))
    _perturb_zero_convs(model)
    model(torch.randn(2, 2, 16, 16), torch.tensor([3, 7])).square().mean().backward()

    trunk = [
        p for n, p in model.unet.named_parameters() if not n.startswith(("conv_in", "conv_out"))
    ]
    assert trunk and all(not p.requires_grad and p.grad is None for p in trunk)
    assert model.unet.conv_in.weight.grad is not None
    assert model.null_context.grad is not None
    assert any(p.grad is not None for p in model.controlnet.parameters())


def test_pretrained_weights_are_loaded(sd_dir):
    """from_pretrained weights reach the trunk; the re-sized conv_in sums the pretrained filter."""
    reference = load_sd_unet(str(sd_dir))
    model = PretrainedSDAdapterGenerator(
        in_channels=1, out_channels=1, base_model_path=str(sd_dir), adapter_type="concat"
    )
    ref_mid = next(reference.mid_block.parameters())
    assert torch.equal(next(model.unet.mid_block.parameters()), ref_mid)
    assert torch.allclose(
        model.unet.conv_in.weight, reference.conv_in.weight.sum(dim=1, keepdim=True)
    )
    assert model.controlnet is None
    assert model(torch.randn(1, 1, 16, 16), torch.tensor([5])).shape == (1, 1, 16, 16)


def test_unknown_adapter_type_raises():
    with pytest.raises(ValueError, match="adapter_type"):
        PretrainedSDAdapterGenerator(adapter_type="lora", unet=_tiny_unet())


def test_controlnet_without_control_channels_raises():
    with pytest.raises(ValueError, match="condition_on_input"):
        PretrainedSDAdapterGenerator(in_channels=1, out_channels=1, unet=_tiny_unet())


def test_undeclared_model_kwarg_raises(sd_dir):
    from spectramr.models.factories.model_factory import ModelFactory

    with pytest.raises(Exception, match="features"):
        ModelFactory().create_generator(
            "pretrained_sd_adapter",
            in_channels=2,
            out_channels=1,
            base_model_path=str(sd_dir),
            features=[32, 64],
            _declared_keys=frozenset({"base_model_path", "features"}),
        )


def test_missing_diffusers_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "diffusers", None)
    with pytest.raises(ImportError, match="diffusers"):
        PretrainedSDAdapterGenerator(base_model_path="unused")
