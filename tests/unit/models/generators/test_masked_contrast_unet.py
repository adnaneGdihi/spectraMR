"""C2 control: a skip-connected U-Net predicting every contrast from any subset, no physics."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.masked_contrast_unet import MaskedContrastUNet


def _net() -> MaskedContrastUNet:
    torch.manual_seed(0)
    net = MaskedContrastUNet(base_channels=8)
    torch.nn.init.normal_(net.head.weight, std=0.5)
    return net.eval()


def _triplet() -> torch.Tensor:
    torch.manual_seed(1)
    return 5.0 + 40.0 * torch.rand(2, 3, 16, 16)


def _out(net, x, avail=None) -> torch.Tensor:
    with torch.no_grad():
        maps, renders = net.render_triplet(x, None, avail)
    assert maps == {}  # a physics-free control has no tissue maps
    return torch.cat(renders, dim=1)


def test_a_dropped_contrast_never_reaches_the_predictions() -> None:
    net, x = _net(), _triplet()
    avail = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    moved = x.clone()
    moved[0, 1] = 900.0 * torch.rand_like(moved[0, 1])
    moved[1, 0] = 0.0
    assert torch.equal(_out(net, x, avail), _out(net, moved, avail))


def test_an_available_contrast_does_reach_the_predictions() -> None:
    net, x = _net(), _triplet()
    moved = x.clone()
    moved[:, 2] *= 3.0
    assert not torch.allclose(_out(net, x), _out(net, moved))


def test_predictions_follow_a_global_scale_of_the_input() -> None:
    # The scale is a gauge here too: normalised in, multiplied back out.
    net, x = _net(), _triplet()
    assert torch.allclose(_out(net, 4.0 * x), 4.0 * _out(net, x), rtol=1e-4)


def test_predictions_are_non_negative_magnitudes_of_the_input_shape() -> None:
    out = _out(_net(), _triplet())
    assert out.shape == (2, 3, 16, 16) and float(out.min()) >= 0.0


def test_a_sample_with_no_contrast_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _net().render_triplet(_triplet(), None, torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))


def test_a_bare_forward_points_at_the_strategy_hook() -> None:
    with pytest.raises(RuntimeError, match="render_triplet"):
        _net()(_triplet())


def test_the_signature_names_every_knob_it_reads() -> None:
    # A **kwargs catch-all hid misspelt knobs from the audit's advertised-options check.
    import inspect

    kinds = {p.kind for p in inspect.signature(MaskedContrastUNet.__init__).parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds
