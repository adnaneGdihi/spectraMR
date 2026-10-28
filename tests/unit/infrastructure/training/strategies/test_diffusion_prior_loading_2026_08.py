"""D12#3: a declared prior must be the trained prior, or the run must stop.

``_setup_prior_model`` had two ways to produce a randomly-initialised "prior"
and then log ``✅ Prior model loaded and frozen``:

1. a ``checkpoint_path`` that is not on disk -> warning, random weights;
2. ``load_state_dict(..., strict=False)`` with ZERO overlapping keys -> an
   architecture mismatch accepted as a successful load.

Either one silently turns a prior-conditioned arm into an unconditioned one and
publishes it as the former (non-negotiable 3). Corpus check before adding the
raises: **0 of 647** ``experiments/inprogress`` arms set ``prior_loading`` at
all; the single repo arm that enables it
(``experiments/training/diffusion/experiment_16_trellis_prior_diffusion.yaml``)
points at ``experiments/results/experiment_6/checkpoints/best_model.pth``, which
does not exist -- i.e. the one arm exercising this path is exactly the one that
was silently running with a random prior.

The strategy is driven through the unbound method with a stand-in ``self``:
constructing a real ``DiffusionTrainingStrategy`` pulls the whole training
environment, and none of it participates in the behaviour under test.
"""

import types

import pytest
import torch

from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)
from tests.fixtures.run_state_files import write_run_state


class _Logging:
    def __init__(self):
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def log_info(self, msg):
        self.infos.append(str(msg))

    def log_warning(self, msg):
        self.warnings.append(str(msg))

    def log_error(self, msg):
        self.errors.append(str(msg))


def _config(checkpoint_path: str):
    prior = types.SimpleNamespace(enabled=True, source="unet", checkpoint_path=checkpoint_path)
    return types.SimpleNamespace(
        data=types.SimpleNamespace(prior_loading=prior),
        model=types.SimpleNamespace(in_channels=1, out_channels=1),
    )


class _Strategy:
    """Minimal stand-in carrying only what ``_setup_prior_model`` touches."""

    def __init__(self, checkpoint_path: str):
        self.config = _config(checkpoint_path)
        self.device = torch.device("cpu")
        self.logging_service = _Logging()
        self.prior_model = None


def _run(strategy):
    return DiffusionTrainingStrategy._setup_prior_model(strategy)


def _stub_builder(monkeypatch, model: torch.nn.Module):
    """Replace the leaf GeneratorBuilder with one that yields ``model``."""

    class _Builder:
        def __init__(self, *a, **k):
            pass

        def with_architecture(self, *a, **k):
            return self

        with_input_channels = with_output_channels = with_architecture

        def validate(self):
            return self

        def build(self):
            return model

    import spectramr.infrastructure.builders.leaf.model_builders as leaf

    monkeypatch.setattr(leaf, "GeneratorBuilder", _Builder)


def test_missing_checkpoint_raises_instead_of_using_random_weights(monkeypatch, tmp_path):
    _stub_builder(monkeypatch, torch.nn.Linear(2, 2))
    strategy = _Strategy(str(tmp_path / "does_not_exist.pth"))

    with pytest.raises(RuntimeError, match="Prior model initialization failed"):
        _run(strategy)

    assert not any("random weights" in w for w in strategy.logging_service.warnings)
    assert not any("loaded and frozen" in i for i in strategy.logging_service.infos)


def test_zero_key_overlap_raises(monkeypatch, tmp_path):
    """A checkpoint of a DIFFERENT architecture loads "successfully" under
    ``strict=False`` while matching nothing."""
    model = torch.nn.Linear(2, 2)
    _stub_builder(monkeypatch, model)
    ckpt = write_run_state(
        tmp_path / "other_arch.pt", {"generator": {"totally.different.weight": torch.zeros(3)}}
    )

    strategy = _Strategy(str(ckpt))
    with pytest.raises(RuntimeError, match=r"shares\s+NO parameter names"):
        _run(strategy)


def test_partial_overlap_is_still_accepted(monkeypatch, tmp_path):
    """``strict=False`` stays: a prior may legitimately load partially. Only the
    zero-overlap case is a mismatch in disguise."""
    model = torch.nn.Linear(2, 2)
    _stub_builder(monkeypatch, model)
    ckpt = write_run_state(tmp_path / "partial.pt", {"generator": {"weight": torch.ones(2, 2)}})

    strategy = _Strategy(str(ckpt))
    _run(strategy)

    assert torch.equal(strategy.prior_model.weight.detach(), torch.ones(2, 2))
    assert any("Matched 1/2" in i for i in strategy.logging_service.infos)
    assert any("loaded and frozen" in i for i in strategy.logging_service.infos)
    assert all(not p.requires_grad for p in strategy.prior_model.parameters())


def test_module_prefix_is_still_stripped(monkeypatch, tmp_path):
    """DataParallel checkpoints keep loading -- the raise must not fire on them."""
    model = torch.nn.Linear(2, 2)
    _stub_builder(monkeypatch, model)
    ckpt = write_run_state(
        tmp_path / "dp.pt",
        {"generator": {"module.weight": torch.ones(2, 2), "module.bias": torch.zeros(2)}},
    )

    strategy = _Strategy(str(ckpt))
    _run(strategy)
    assert any("Matched 2/2" in i for i in strategy.logging_service.infos)


def test_a_pre_v1_prior_names_the_converter(monkeypatch, tmp_path):
    """An old envelope is a schema error, not an architecture mismatch: the user
    must be told to convert the run, not to point at a different checkpoint."""
    _stub_builder(monkeypatch, torch.nn.Linear(2, 2))
    ckpt = tmp_path / "old.pt"
    torch.save({"model_state_dict": torch.nn.Linear(2, 2).state_dict()}, ckpt)

    with pytest.raises(ValueError, match="convert_checkpoints_v1") as info:
        _run(_Strategy(str(ckpt)))
    assert "NO parameter names" not in str(info.value)


def test_disabled_prior_loading_is_a_no_op(monkeypatch):
    strategy = _Strategy("")
    strategy.config.data.prior_loading.enabled = False
    _run(strategy)
    assert strategy.prior_model is None


def test_zero_overlap_message_keeps_the_key_counts(monkeypatch, tmp_path):
    _stub_builder(monkeypatch, torch.nn.Linear(2, 2))
    ckpt = write_run_state(
        tmp_path / "alien.pt",
        {"generator": {"a.w": torch.zeros(1), "b.w": torch.zeros(1), "c.w": torch.zeros(1)}},
    )
    with pytest.raises(RuntimeError, match=r"2 expected keys, 3 unexpected in the checkpoint"):
        _run(_Strategy(str(ckpt)))


def test_an_unrelated_codec_error_keeps_its_own_message(monkeypatch, tmp_path):
    """Only a genuine zero overlap is reported as an architecture mismatch."""
    import spectramr.infrastructure.training.strategies.diffusion as diffusion_mod

    _stub_builder(monkeypatch, torch.nn.Linear(2, 2))

    def _boom(*args, **kwargs):
        raise ValueError("checkpoint x.pt has unknown part(s) ['bogus']")

    monkeypatch.setattr(diffusion_mod, "weights", _boom)
    with pytest.raises(ValueError, match="unknown part") as info:
        _run(_Strategy(str(tmp_path / "x.pt")))
    assert "NO parameter names" not in str(info.value)
