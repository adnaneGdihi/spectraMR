"""Contrast forwarding from the reconstruction strategy to contrast-conditioned generators (#2410)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from spectramr.infrastructure.training.strategies.mixins.contrast_forwarding import (
    align_contrast_rows,
    bloch_loss_context,
    build_acquisition_lookup,
    names_parameter,
)
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.models.losses.computers.unified_diffusion_reconstruction import _call_safe_loss
from spectramr.models.losses.registry import create_loss
from spectramr.models.registry import get_model_class

_ROWS = [
    {
        "name": "T1",
        "TE": 18.4,
        "TR": 500.0,
        "TI": -1.0,
        "FA": 90.0,
        "B0": 0.3,
        "contrast_type": "spin_echo",
    },
    {
        "name": "T2",
        "TE": 128.0,
        "TR": 5500.0,
        "TI": -1.0,
        "FA": 90.0,
        "B0": 0.3,
        "contrast_type": "spin_echo",
    },
    {
        "name": "FLAIR",
        "TE": 98.0,
        "TR": 7500.0,
        "TI": 1655.0,
        "FA": 90.0,
        "B0": 0.3,
        "contrast_type": "inversion_recovery",
    },
]
_MAP = {"T1": 0, "T2": 1, "FLAIR": 2}


class _Generator(torch.nn.Module):
    def forward(self, x, acquisition_params=None, contrast_idx=None):
        return x


class _Alias(torch.nn.Module):
    def forward(self, x, **kwargs):
        return x


def _strategy(enabled: bool = True, rows: list[dict[str, Any]] | None = None) -> Any:
    params = None if rows is None else [SimpleNamespace(model_dump=lambda r=r: r) for r in rows]
    s = object.__new__(ReconstructionTrainingStrategy)
    s.config = SimpleNamespace(
        data=SimpleNamespace(
            multi_contrast=SimpleNamespace(
                enabled=enabled, contrast_map=_MAP, acquisition_params=params
            )
        )
    )
    return s


def test_names_parameter_ignores_var_keyword() -> None:
    assert names_parameter(_Generator().forward, "contrast_idx")
    assert not names_parameter(_Alias().forward, "contrast_idx")


def test_strategy_forwards_contrast_idx_and_acquisition_rows() -> None:
    ids = torch.tensor([2, 0])
    kwargs = _strategy(rows=_ROWS)._contrast_idx_kwargs(_Generator().forward, ids, 2)
    assert torch.equal(kwargs["contrast_idx"], ids)
    acq = kwargs["acquisition_params"]
    assert acq["TR"].tolist() == [7500.0, 500.0]
    assert acq["TI"].tolist() == [1655.0, -1.0]
    assert acq["B0"].tolist() == pytest.approx([0.3, 0.3])
    assert acq["contrast_type"].tolist() == [1, 0]


def test_alias_with_bare_kwargs_receives_nothing() -> None:
    kwargs = _strategy(rows=_ROWS)._contrast_idx_kwargs(_Alias().forward, torch.tensor([0]), 1)
    assert kwargs == {}


def test_missing_contrast_id_raises_when_multi_contrast_enabled() -> None:
    with pytest.raises(ValueError, match="no contrast id"):
        _strategy()._contrast_idx_kwargs(_Generator().forward, None, 2)


def test_disabled_multi_contrast_forwards_nothing() -> None:
    """The control arm: the same network, unconditioned."""
    kwargs = _strategy(enabled=False, rows=_ROWS)._contrast_idx_kwargs(
        _Generator().forward, torch.tensor([0]), 1
    )
    assert kwargs == {}


def test_rows_expand_over_flattened_slices() -> None:
    assert align_contrast_rows(torch.tensor([1, 2]), 4).tolist() == [1, 1, 2, 2]
    with pytest.raises(ValueError):
        align_contrast_rows(torch.tensor([1, 2]), 3)


def test_lookup_refuses_rows_that_do_not_match_the_map() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        build_acquisition_lookup(_MAP, _ROWS[:2])
    with pytest.raises(ValueError, match=r"0\.\.2"):
        build_acquisition_lookup({"T1": 0, "T2": 1, "FLAIR": 3}, _ROWS)


@pytest.mark.parametrize(
    ("module", "cls"),
    [
        ("hypernetwork_contrast_generator", "HypernetworkContrastGenerator"),
        ("siren_universal", "SIRENUniversalGenerator"),
        ("bloch_parameter_bottleneck", "BlochParameterBottleneckGenerator"),
        ("counterfactual_healthy_generator", "CounterfactualHealthyGenerator"),
        ("cross_contrast_cross_domain_generator", "CrossContrastCrossDomainGenerator"),
    ],
)
def test_conditioned_generators_name_both_kwargs(module: str, cls: str) -> None:
    mod = __import__(f"spectramr.models.generators.{module}", fromlist=[cls])
    params = inspect.signature(getattr(mod, cls).forward).parameters
    assert "contrast_idx" in params and "acquisition_params" in params


def test_prepare_generator_inputs_reads_either_contrast_spelling() -> None:
    """The live seam: ``contrast_idx`` in the batch context reaches the forward."""
    s = _strategy(rows=_ROWS)
    s.env = SimpleNamespace(generator=_Generator())
    s._needs_kspace_adjoint = lambda ctx: False
    s._is_implicit_model = lambda: False
    s.config.model = SimpleNamespace(in_channels=1)
    s.config.data.multislice_enabled = False
    x = torch.rand(2, 1, 8, 8)
    ctx = {"use_dc": False, "hr": x, "contrast_idx": torch.tensor([1, 2])}
    _, kwargs = s._prepare_generator_inputs(ctx, x)
    assert kwargs["contrast_idx"].tolist() == [1, 2]
    assert kwargs["acquisition_params"]["TE"].tolist() == [128.0, 98.0]


def test_bloch_loss_context_is_empty_for_a_plain_generator() -> None:
    assert bloch_loss_context(_Generator()) == {}


def test_bloch_bottleneck_refuses_a_forward_without_protocol() -> None:
    model = get_model_class("bloch_parameter_bottleneck")(base_channels=4)
    with pytest.raises(ValueError, match="acquisition_params"):
        model(torch.rand(1, 1, 16, 16))


def test_bloch_bottleneck_renders_the_batch_contrast_and_feeds_bloch_consistency() -> None:
    """Reachability: registry model -> strategy seam -> Bloch render -> declared loss."""
    torch.manual_seed(0)
    model = get_model_class("bloch_parameter_bottleneck")(base_channels=4)
    s = _strategy(rows=_ROWS)
    s.env = SimpleNamespace(generator=model)
    s._needs_kspace_adjoint = lambda ctx: False
    s._is_implicit_model = lambda: False
    s.config.model = SimpleNamespace(in_channels=1)
    s.config.data.multislice_enabled = False
    x = torch.rand(2, 1, 16, 16)
    ctx = {"use_dc": False, "hr": x, "contrast_idx": torch.tensor([2, 0])}
    lr, kwargs = s._prepare_generator_inputs(ctx, x)
    pred = model(lr, **kwargs)
    assert pred.shape == x.shape
    assert model.last_acquisition_params["contrast_type"].tolist() == [1, 0]

    loss_kwargs = bloch_loss_context(model)
    assert loss_kwargs["context"]["tissue_params"] is model.last_tissue_params
    loss = _call_safe_loss(create_loss("bloch_consistency", norm="l1"), pred, x, **loss_kwargs)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.parameter_net.head.weight.grad is not None


class _CapturedError(Exception):
    pass


def test_compute_losses_hands_the_loss_computer_the_bloch_context() -> None:
    """The strategy step itself puts the generator's maps and protocol in the loss context."""
    model = get_model_class("bloch_parameter_bottleneck")(base_channels=4)
    x = torch.rand(2, 1, 16, 16)
    acq = build_acquisition_lookup(_MAP, _ROWS)(torch.tensor([1, 2]))
    pred = model(x, acquisition_params=acq)
    s = _strategy(rows=_ROWS)
    s.env = SimpleNamespace(generator=model, losses={})
    s.config.training = SimpleNamespace(tall_sr=None)
    s.config.model = SimpleNamespace(model_type="bloch_parameter_bottleneck")
    s.pinn_module = None
    s._prepare_batch_context = lambda i, t, **kw: {"hr": t}
    s._prepare_generator_inputs = lambda ctx, i: (i, {})
    s._generate_predictions = lambda lr, kw, ctx: (pred, None)
    seen: dict[str, Any] = {}

    def _compute(**kwargs: Any) -> Any:
        seen.update(kwargs)
        raise _CapturedError

    s.loss_computer = SimpleNamespace(compute=_compute)
    with pytest.raises(_CapturedError):
        s._compute_losses_impl(x, x, epoch=0)
    assert seen["context"]["tissue_params"] is model.last_tissue_params
    assert seen["context"]["acquisition_params"] is model.last_acquisition_params
