"""``freeze_vae`` holds on every build, not only the warm-starting one.

Before the checkpointing switch-over the LDM constructor loaded and froze the stage-1
VAE whenever ``vae_checkpoint_path`` was set, so ``make_model(checkpoint=...)`` handed
``fit`` a model whose autoencoder did not train. A build that loads its own checkpoint
afterwards skips the warm-start, and must still freeze.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from spectramr.infrastructure.training.builders.model_builder import ModelBuilder
from spectramr.models.generators.latent_diffusion_generator import LatentDiffusionGenerator


class _Host(torch.nn.Module):
    """Stands in for the LDM, running the real ``load_autoencoder_state``."""

    def __init__(self) -> None:
        super().__init__()
        self.autoencoder = torch.nn.Linear(2, 2)
        self.unet = torch.nn.Linear(2, 2)
        self.calls: list[dict] = []

    def load_autoencoder_state(self, state, *, freeze, source):
        self.calls.append({"freeze": freeze, "source": source})
        LatentDiffusionGenerator.load_autoencoder_state(self, state, freeze=freeze, source=source)


def _builder(**model_kwargs) -> ModelBuilder:
    builder = ModelBuilder.__new__(ModelBuilder)
    builder._config = SimpleNamespace(
        model=SimpleNamespace(checkpoint_path=None, model_kwargs=model_kwargs)
    )
    builder._models = {}
    builder._device = torch.device("cpu")
    return builder


def test_a_build_without_warm_start_still_freezes_the_stage1_vae(tmp_path) -> None:
    """No file is read: the run's own checkpoint supplies the weights next."""
    host = _Host()
    before = {k: v.clone() for k, v in host.autoencoder.state_dict().items()}
    _builder(vae_checkpoint_path=str(tmp_path / "absent.pt"))._freeze_stage1_vae(host)

    assert [c["freeze"] for c in host.calls] == [True]
    assert not any(p.requires_grad for p in host.autoencoder.parameters())
    assert all(p.requires_grad for p in host.unet.parameters())
    for key, value in host.autoencoder.state_dict().items():
        assert torch.equal(value, before[key])


def test_freeze_vae_false_leaves_the_autoencoder_trainable(tmp_path) -> None:
    host = _Host()
    _builder(vae_checkpoint_path=str(tmp_path / "absent.pt"), freeze_vae=False)._freeze_stage1_vae(
        host
    )
    assert host.calls == []
    assert all(p.requires_grad for p in host.autoencoder.parameters())


def test_no_stage1_vae_means_nothing_to_freeze() -> None:
    host = _Host()
    _builder()._freeze_stage1_vae(host)
    assert host.calls == []


def test_build_generator_freezes_whenever_it_does_not_warm_start() -> None:
    """Before ``train()``, so the frozen autoencoder ends in train mode as it did when
    the constructor froze it."""
    src = inspect.getsource(ModelBuilder.build_generator)
    branch = "else:\n                self._freeze_stage1_vae(generator)"
    assert branch in src
    assert src.index("self._warm_start(generator)") < src.index(branch)
    assert src.index(branch) < src.index("generator.train()")
