"""``unrolled_tissue_net``: learned gradient descent on (rho, T1, T2) through the TSE render."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.unrolled_tissue_net import UnrolledTissueNet

#: M4Raw header census (DESIGN_tissue_map_cohort.md §1), [T1, T2, FLAIR].
PROTOCOL = [
    {"family": 0, "tr": 500.0, "te": 18.4, "ti": 0.0, "esp": 18.4, "etl": 1, "refocus_deg": 180.0},
    {
        "family": 0,
        "tr": 5500.0,
        "te": 128.0,
        "ti": 0.0,
        "esp": 16.0,
        "etl": 15,
        "refocus_deg": 160.0,
    },
    {
        "family": 1,
        "tr": 7500.0,
        "te": 98.0,
        "ti": 1655.0,
        "esp": 16.3,
        "etl": 11,
        "refocus_deg": 160.0,
    },
]


def _net(active: bool = True, **kw) -> UnrolledTissueNet:
    torch.manual_seed(0)
    net = UnrolledTissueNet(base_channels=8, iterations=2, **kw)
    if active:  # the head starts at zero; give it weights so the residual path is read
        torch.nn.init.normal_(net.prior.head.weight, std=0.5)
    return net.eval()


def _triplet(b: int = 2, h: int = 16) -> torch.Tensor:
    torch.manual_seed(1)
    return 5.0 + 40.0 * torch.rand(b, 3, h, h)  # noise units, most voxels above the floor


def _maps(net, x, avail=None):
    with torch.no_grad():
        maps, _ = net.unroll(x, PROTOCOL, avail)
    return torch.cat([maps["rho"], maps["t1"], maps["t2"]], dim=1)


def test_a_dropped_contrast_never_reaches_the_maps() -> None:
    # Leave-one-contrast-out is only a test if the held-out image cannot leak in, through
    # the input, the fitted gain or the residual.
    net, x = _net(), _triplet()
    avail = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    perturbed = x.clone()
    perturbed[0, 1] = 1000.0 * torch.rand_like(perturbed[0, 1])
    perturbed[1, 2] = 0.0
    assert torch.equal(_maps(net, x, avail), _maps(net, perturbed, avail))


def test_an_available_contrast_does_reach_the_maps() -> None:
    # The planted counterpart: the leak test must fail for a contrast that is acquired.
    net, x = _net(), _triplet()
    perturbed = x.clone()
    perturbed[:, 1] *= 3.0
    assert not torch.allclose(_maps(net, x), _maps(net, perturbed))


def test_the_maps_do_not_depend_on_the_scanner_scale() -> None:
    # The closed-form gain and the input normalisation make a global per-image scale a
    # gauge the maps cannot see.
    net, x = _net(), _triplet()
    assert torch.allclose(_maps(net, x), _maps(net, 7.5 * x), rtol=1e-4, atol=1e-4)


def test_an_untrained_prior_makes_no_update() -> None:
    net, x = _net(active=False), _triplet()
    net.iterations = 1
    one = _maps(net, x)
    net.iterations = 2
    net.step_embedding = torch.nn.Parameter(torch.zeros(2))
    assert torch.allclose(one, _maps(net, x))


def test_maps_stay_inside_their_bounds_and_renders_match_the_input() -> None:
    net, x = _net(), _triplet()
    with torch.no_grad():
        maps, renders = net.unroll(x, PROTOCOL)
    for name, (lo, hi) in {"rho": (0.0, 1.0), "t1": (50.0, 5000.0), "t2": (5.0, 2000.0)}.items():
        assert float(maps[name].min()) >= lo and float(maps[name].max()) <= hi, name
    assert [tuple(r.shape) for r in renders] == [(2, 1, 16, 16)] * 3
    assert len(net.last_residual_norms) == 2


def test_gradients_reach_every_part_through_the_checkpointed_unroll() -> None:
    net, x = _net(), _triplet()
    net.train()
    maps, renders = net.unroll(x, PROTOCOL)
    (sum(r.mean() for r in renders) + maps["t1"].mean()).backward()
    for name in ("init_net.1.weight", "prior.head.weight", "prior.enc.0.weight"):
        grad = dict(net.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def test_a_sample_with_no_contrast_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _net().unroll(_triplet(), PROTOCOL, torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))


def test_the_protocol_must_cover_the_three_contrasts() -> None:
    with pytest.raises(ValueError, match="one protocol per contrast"):
        _net().unroll(_triplet(), PROTOCOL[:2])


def test_a_bare_forward_refuses_rather_than_render_without_a_protocol() -> None:
    with pytest.raises(RuntimeError, match="render_triplet"):
        _net()(_triplet())


_B1 = {"initializer": "contrast_transformer",
       "initializer_kwargs": {"dim": 16, "depth": 1, "heads": 2, "window": 4}}


def test_the_b1_initialiser_keeps_a_dropped_contrast_out_of_the_maps() -> None:
    net, x = _net(**_B1), _triplet()
    avail = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    moved = x.clone()
    moved[0, 1] = 1000.0 * torch.rand_like(moved[0, 1])
    moved[1, 2] = 0.0
    assert torch.equal(_maps(net, x, avail), _maps(net, moved, avail))


def test_the_b1_initialiser_keeps_the_maps_scale_invariant() -> None:
    net, x = _net(**_B1), _triplet()
    assert torch.allclose(_maps(net, x), _maps(net, 7.5 * x), rtol=1e-4, atol=1e-4)


def test_an_unknown_initialiser_is_refused() -> None:
    with pytest.raises(ValueError, match="contrast_transformer"):
        UnrolledTissueNet(initializer="vit")


def test_the_signature_names_every_knob_it_reads() -> None:
    # A **kwargs catch-all hid misspelt knobs from the audit: `initialiser:` built the conv
    # initialiser, and B1 trained as plain B2.
    import inspect

    kinds = {p.kind for p in inspect.signature(UnrolledTissueNet.__init__).parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds
