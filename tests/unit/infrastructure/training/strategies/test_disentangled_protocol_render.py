"""The echo-train render uses each contrast's declared TR, TE and TI (#2618).

Before, the spin-echo and inversion-recovery renders read only TI: every row was an echo train
of 16 echoes 10 ms apart read at echo 8, with no repetition time, so a T1-weighted row (TR 500,
TE 18.4) and a T2-weighted row (TR 5500, TE 128) rendered identically. With the protocol
declared, the render of an ideal CPMG train is ``rho * (1 - exp(-TR/T1)) * exp(-TE/T2)`` for
spin echo and ``rho * |1 - 2 exp(-TI/T1) + exp(-TR/T1)| * exp(-TE/T2)`` for inversion recovery.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.fno_epg import FNOEPGSurrogate
from spectramr.infrastructure.training.strategies.disentangled_bloch_inputs import (
    check_protocol_matches_families,
    parse_protocol_map,
    parse_sequence_map,
)
from spectramr.infrastructure.training.strategies.disentangled_strategy import (
    DisentangledTrainingStrategy,
)

PROTOCOL = {
    "T1": {"TR": 500.0, "TE": 18.4},
    "T2": {"TR": 5500.0, "TE": 128.0},
    "FLAIR": {"TR": 7500.0, "TE": 98.0, "TI": 1655.0},
}
FAMILIES = {"T1": "SE", "T2": "SE", "FLAIR": "IR"}

#: Two voxels: a white-matter-like tissue and a cerebrospinal-fluid-like one.
WM = {"rho": 0.7, "t1": 800.0, "t2": 80.0}
CSF = {"rho": 1.0, "t1": 4000.0, "t2": 2000.0}


def _strategy(protocol: dict | None = PROTOCOL) -> DisentangledTrainingStrategy:
    s = object.__new__(DisentangledTrainingStrategy)
    s._sequence_map = parse_sequence_map(FAMILIES)
    s._protocol_map = parse_protocol_map(protocol) if protocol else {}
    s._fno_epg = None
    s._in_channels = 1
    s._dipole_pad_z = 0
    return s


def _tissue(n: int = 1, *, predicted_te: float = 35.0) -> dict:
    """``n`` rows of the two-voxel phantom (shape ``(n, 1, 1, 2)``)."""
    t = {k: torch.tensor([[[[WM[k], CSF[k]]]]]).repeat(n, 1, 1, 1) for k in ("rho", "t1", "t2")}
    t["b1_plus"] = torch.ones(n, 1, 1, 2)
    # what the style code predicts: the declared protocol must win over these
    t.update(
        {
            f"predicted_{p}": torch.full((n, 1, 1, 1), v)
            for p, v in (("tr", 700.0), ("te", predicted_te), ("ti", 690.0), ("alpha", 10.0))
        }
    )
    return t


def _signal(strategy: DisentangledTrainingStrategy, contrast: str, **kw) -> torch.Tensor:
    """The render mapped back from [-1, 1] to [0, 1] (it is divided by its own maximum)."""
    out = strategy._synthesize_via_bloch(_tissue(**kw), {"contrast": [contrast]}, "a")
    return ((out + 1.0) / 2.0).view(-1)


def _expected(contrast: str, voxel: dict) -> float:
    p = PROTOCOL[contrast]
    sat = math.exp(-p["TR"] / voxel["t1"])
    if FAMILIES[contrast] == "IR":
        weight = abs(1.0 - 2.0 * math.exp(-p["TI"] / voxel["t1"]) + sat)
    else:
        weight = 1.0 - sat
    return voxel["rho"] * weight * math.exp(-p["TE"] / voxel["t2"])


@pytest.mark.parametrize("contrast", ["T1", "T2", "FLAIR"])
def test_the_render_follows_the_closed_form_of_the_declared_protocol(contrast: str) -> None:
    got = _signal(_strategy(), contrast)
    want = torch.tensor([_expected(contrast, WM), _expected(contrast, CSF)])
    assert torch.allclose(got, want / want.max(), atol=2e-3), (got, want / want.max())


def test_t1w_and_t2w_now_order_the_tissues_oppositely() -> None:
    """CSF is dark on a short-TR image and bright on a long-TE one: the contrast the old
    render could not make, since both rows shared one echo train."""
    s = _strategy()
    t1w, t2w = _signal(s, "T1"), _signal(s, "T2")
    assert t1w[1] < t1w[0]  # CSF darker than white matter
    assert t2w[1] > t2w[0]  # CSF brighter than white matter


def test_without_a_declared_protocol_t1w_and_t2w_share_one_render() -> None:
    """The old behaviour, kept for arms that declare no protocol (and the reason to declare one)."""
    s = _strategy(protocol=None)
    assert torch.allclose(_signal(s, "T1"), _signal(s, "T2"))


def test_the_declared_protocol_wins_over_the_style_codes_prediction() -> None:
    s = _strategy()
    assert torch.allclose(_signal(s, "T2", predicted_te=5.0), _signal(s, "T2", predicted_te=300.0))


@pytest.mark.parametrize("protocol", [PROTOCOL, None])
@pytest.mark.parametrize("contrast", ["T1", "T2", "FLAIR"])
def test_a_spin_echo_refocuses_static_dephasing_so_chi_leaves_echo_trains_alone(
    contrast: str, protocol
) -> None:
    """R2' = gamma |dB0| from a susceptibility map decays a gradient echo, not a spin echo.

    The chi term multiplied every row by exp(-R2' TE). With the declared TE that is about
    7x stronger on T2w (128 ms) than on T1w (18.4 ms) for one shared chi map, which pushed
    the map toward 0 or made the T2 maps compensate.
    """
    s = _strategy(protocol)
    # An 8x8 white-matter patch: the dipole field of a 1x2 map is ~0 and would hide the term.
    tissue = {k: torch.full((1, 1, 8, 8), WM[k]) for k in ("rho", "t1", "t2")}
    tissue["rho"] = tissue["rho"] * torch.linspace(0.5, 1.0, 64).view(1, 1, 8, 8)
    tissue["b1_plus"] = torch.ones(1, 1, 8, 8)
    tissue.update({f"predicted_{p}": torch.full((1, 1, 1, 1), v) for p, v in
                   (("tr", 700.0), ("te", 35.0), ("ti", 690.0), ("alpha", 10.0))})  # fmt: skip
    plain = s._synthesize_via_bloch(dict(tissue), {"contrast": [contrast]}, "a")
    gen = torch.Generator().manual_seed(0)
    tissue["chi"] = 2.0 * torch.randn(1, 1, 8, 8, generator=gen)  # O(1) logits, as emitted
    with_chi = s._synthesize_via_bloch(dict(tissue), {"contrast": [contrast]}, "a")
    assert torch.allclose(with_chi, plain, atol=1e-6)


def test_a_gradient_echo_row_is_still_dephased_by_chi() -> None:
    """The control: the term itself is kept, on the rows whose physics has it."""
    from spectramr.infrastructure.physics.multi_physics_bloch import MultiPhysicsBlochLayer

    s = object.__new__(DisentangledTrainingStrategy)
    s._sequence_map = parse_sequence_map({"T1": "GRE"})
    s._protocol_map = {}
    s._fno_epg = None
    s._in_channels = 1
    s._dipole_pad_z = 0
    s._bloch_layer = MultiPhysicsBlochLayer(learnable_flip_angle=False)
    tissue = {k: torch.full((1, 1, 8, 8), WM[k]) for k in ("rho", "t1", "t2")}
    tissue["rho"] = tissue["rho"] * torch.linspace(0.5, 1.0, 64).view(1, 1, 8, 8)
    tissue["t2_star"] = torch.full((1, 1, 8, 8), 50.0)
    tissue.update({f"predicted_{p}": torch.full((1, 1, 1, 1), v) for p, v in
                   (("tr", 30.0), ("te", 20.0), ("ti", 0.0), ("alpha", 20.0))})  # fmt: skip
    plain = s._synthesize_via_bloch(dict(tissue), {"contrast": ["T1"]}, "a")
    tissue["chi"] = 2.0 * torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(0))
    with_chi = s._synthesize_via_bloch(dict(tissue), {"contrast": ["T1"]}, "a")
    assert not torch.allclose(with_chi, plain, atol=1e-3)


def test_a_declared_protocol_needs_no_predicted_parameters() -> None:
    s = _strategy()
    tissue = {k: v for k, v in _tissue().items() if not k.startswith("predicted_")}
    out = s._synthesize_via_bloch(tissue, {"contrast": ["T2"]}, "a")
    assert torch.isfinite(out).all()


def test_rows_without_an_echo_train_render_refuse_the_protocol() -> None:
    s = _strategy()
    tissue = {k: v for k, v in _tissue().items() if k != "b1_plus"}  # no Path 2, no FNO
    with pytest.raises(ValueError, match="no echo-train render"):
        s._synthesize_via_bloch(tissue, {"contrast": ["T2"]}, "a")


def test_the_render_is_differentiable_in_the_tissue_maps() -> None:
    s = _strategy()
    tissue = _tissue()
    leaves = [tissue[k].requires_grad_(True) for k in ("rho", "t1", "t2")]
    s._synthesize_via_bloch(tissue, {"contrast": ["T2"]}, "a").sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in leaves)


def test_validation_hands_the_render_its_rows_acquisition_through_the_batch() -> None:
    s = _strategy()
    acq = {k: torch.tensor([[[[v]]]]) for k, v in (("TR", 5500.0), ("TE", 128.0), ("TI", 0.0))}
    chunk = {"contrast_type": torch.tensor([[[[0]]]]), "acquisition": acq}
    via_chunk = ((s._synthesize_via_bloch(_tissue(), chunk, "a") + 1.0) / 2.0).view(-1)
    assert torch.allclose(via_chunk, _signal(s, "T2"), atol=1e-6)


def test_the_arms_protocol_agrees_with_its_families() -> None:
    check_protocol_matches_families(parse_protocol_map(PROTOCOL), parse_sequence_map(FAMILIES))


class TestFnoSurrogateTakesPerRowEchoSpacing:
    def _inputs(self, n: int = 2):
        rho = torch.rand(n, 1, 4, 4) + 0.2
        return rho, torch.full_like(rho, 1000.0), torch.full_like(rho, 80.0), torch.ones_like(rho)

    def test_a_uniform_tensor_equals_the_float(self) -> None:
        rho, t1, t2, b1 = self._inputs()
        torch.manual_seed(0)
        net = FNOEPGSurrogate(tissue_channels=4, hidden_width=8, n_layers=1, max_etl=16)
        as_float = net(rho, t1, t2, b1, te_spacing=2.3, echo_train_length=16)
        as_tensor = net(
            rho, t1, t2, b1, te_spacing=torch.full((2, 1, 1, 1), 2.3), echo_train_length=16
        )
        assert torch.allclose(as_float, as_tensor, atol=1e-6)

    def test_the_echo_spacing_reaches_the_film_conditioning_per_row(self) -> None:
        rho, t1, t2, b1 = self._inputs()
        torch.manual_seed(0)
        net = FNOEPGSurrogate(tissue_channels=4, hidden_width=8, n_layers=1, max_etl=16)
        for p in net.parameters():  # make the network sensitive: the default init is near-inert
            torch.nn.init.normal_(p, std=0.5)
        spacing = torch.tensor([2.3, 16.0]).view(2, 1, 1, 1)
        both = net(rho, t1, t2, b1, te_spacing=spacing, echo_train_length=16)
        a = net(rho[:1], t1[:1], t2[:1], b1[:1], te_spacing=2.3, echo_train_length=16)
        b = net(rho[1:], t1[1:], t2[1:], b1[1:], te_spacing=16.0, echo_train_length=16)
        assert torch.allclose(both[:1], a, atol=1e-5)
        assert torch.allclose(both[1:], b, atol=1e-5)
