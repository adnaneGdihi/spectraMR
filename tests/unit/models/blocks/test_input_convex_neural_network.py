"""Tests for InputConvexNeuralNetwork (B-1.5/3.9 Brenier potential primitive, T1 #20)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spectramr.models.blocks.input_convex_neural_network import InputConvexNeuralNetwork


def _net(**kw) -> InputConvexNeuralNetwork:
    return InputConvexNeuralNetwork(hidden_channels=16, n_layers=3, **kw)


def test_potential_is_midpoint_convex() -> None:
    # THE structural property (Brenier): psi((a+b)/2) <= (psi(a)+psi(b))/2. This is genuine
    # convexity (NOT monotonicity — a convex function can decrease then increase).
    torch.manual_seed(0)
    m = _net().eval()
    a, b = torch.rand(8, 1, 16, 16), torch.rand(8, 1, 16, 16)
    pa, pb, pm = m.potential(a), m.potential(b), m.potential((a + b) / 2)
    assert float((pm - 0.5 * (pa + pb)).detach().max()) < 1e-3


def test_brenier_map_is_monotone() -> None:
    # The gradient of a (strongly) convex potential is a monotone map:
    # <T(a)-T(b), a-b> >= 0 — the defining property of a valid OT map.
    torch.manual_seed(0)
    m = _net().eval()
    a, b = torch.rand(6, 1, 16, 16), torch.rand(6, 1, 16, 16)
    ta, tb = m.brenier_map(a), m.brenier_map(b)
    mono = ((ta - tb) * (a - b)).flatten(1).sum(1).detach()
    assert float(mono.min()) >= -1e-3


def test_near_identity_init() -> None:
    # Displacement form (T = x + map_scale*grad psi) with small map_scale -> T(x) ~ x at init,
    # a sane OT starting point (and keeps the rendered image in range early).
    torch.manual_seed(0)
    m = _net().eval()
    x = torch.rand(4, 1, 16, 16)
    t = m.brenier_map(x).detach()
    assert t.shape == (4, 1, 16, 16)
    assert float((t - x).norm() / x.norm()) < 0.5


def test_enforce_convexity_toggles_the_softplus_reparam() -> None:
    # The one-knob ablation mechanism: enforce_convexity=True -> convex-path weights are
    # softplus(raw) (>= 0, structural convexity guarantee); False -> raw (sign-free, no
    # guarantee). Tested directly on _zw so it is robust (not a fragile convexity-breaking probe).
    raw = torch.tensor([[-2.0, 0.5], [1.0, -3.0]])
    on = _net(enforce_convexity=True)._zw(raw)
    off = _net(enforce_convexity=False)._zw(raw)
    assert float(on.min()) >= 0.0  # softplus -> all non-negative
    assert float(off.min()) < 0.0  # raw passthrough -> negatives survive
    assert torch.allclose(off, raw)


def test_convex_by_construction_even_with_negative_raw_weights() -> None:
    # enforce_convexity=True must stay convex for ANY raw weights (structural, not learned):
    # the softplus reparam makes the convex path non-negative regardless.
    torch.manual_seed(1)
    m = _net(enforce_convexity=True)
    with torch.no_grad():
        for p in m.z_raw:
            p.normal_(-2.0, 1.5)
        m.z_final_raw.normal_(-2.0, 1.5)
    m.eval()
    a, b = torch.rand(8, 1, 16, 16), torch.rand(8, 1, 16, 16)
    pa, pb, pm = m.potential(a), m.potential(b), m.potential((a + b) / 2)
    assert float((pm - 0.5 * (pa + pb)).detach().max()) < 1e-3


def test_field_conditioned_potential_stays_convex() -> None:
    # B-1.5 field conditioning (gamma>=0, beta) must preserve convexity in x even with a
    # non-trivial field MLP.
    torch.manual_seed(0)
    m = _net(field_dim=1)
    with torch.no_grad():
        m.field_mlp.weight.normal_(0.0, 0.5)
        m.field_mlp.bias.normal_(0.0, 0.5)
    m.eval()
    a, b = torch.rand(8, 1, 16, 16), torch.rand(8, 1, 16, 16)
    fld = torch.rand(8, 1)
    pa, pb, pm = m.potential(a, fld), m.potential(b, fld), m.potential((a + b) / 2, fld)
    assert float((pm - 0.5 * (pa + pb)).detach().max()) < 1e-3


def test_double_backward_trains_the_map() -> None:
    # Training backprops through grad_x psi (a double backward); the map must be learnable.
    torch.manual_seed(0)
    m = _net()
    m.train()
    x = torch.rand(4, 1, 16, 16)
    tgt = torch.rand(4, 1, 16, 16)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = None
    loss = None
    for _ in range(40):
        opt.zero_grad()
        loss = torch.nn.functional.l1_loss(m.brenier_map(x), tgt)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
    assert loss is not None and first is not None
    assert float(loss.detach()) < first


@pytest.mark.parametrize("enforce", [True, False])
def test_near_identity_at_production_config(enforce: bool) -> None:
    # REGRESSION (review HIGH): "near-identity init" was only tested at the toy config (h=16,
    # n=3); at the PRODUCTION config (h=64, n_layers=4) the un-normalised gradient exploded
    # (~79% clamp saturation). Fan-in normalisation must keep BOTH ablation arms near-identity.
    torch.manual_seed(0)
    m = InputConvexNeuralNetwork(hidden_channels=64, n_layers=4, enforce_convexity=enforce).eval()
    x = torch.rand(2, 1, 64, 64)
    t = m.brenier_map(x).detach()
    assert float((t - x).norm() / x.norm()) < 0.5
    assert float(((t < 0) | (t > 1)).float().mean()) < 0.2  # not saturated


def test_convex_constraint_underperforms_free_fit_expressivity_ceiling() -> None:
    # REGRESSION (review HIGH): the curl-free scalar-potential map cannot express general
    # per-pixel targets; the free (non-convex) map fits them better. Documents the ceiling so
    # a "convex worse" result is understood as the constraint, not a bug. (The OLD test only
    # asserted loss DECREASES, which masked this.)
    torch.manual_seed(0)
    x = torch.rand(4, 1, 32, 32)
    # ANTI-MONOTONE target (T(x)=1-x): a monotone map (gradient of a CONVEX potential)
    # structurally CANNOT represent a decreasing map, while the free (non-convex) map can.
    # This is a robust structural separation, not a training race.
    tgt = 1.0 - x

    def _fit(enforce):
        m = InputConvexNeuralNetwork(hidden_channels=32, n_layers=4, enforce_convexity=enforce)
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        loss = None
        for _ in range(150):
            opt.zero_grad()
            loss = torch.nn.functional.l1_loss(m.brenier_map(x), tgt)
            loss.backward()
            opt.step()
        return float(loss.detach())

    convex_l1, free_l1 = _fit(True), _fit(False)
    assert free_l1 < convex_l1  # convexity forbids the anti-monotone map; the free map fits it


def test_field_dim_requires_n_layers_at_least_3() -> None:
    with pytest.raises(ValueError, match="n_layers>=3"):
        InputConvexNeuralNetwork(field_dim=1, n_layers=2)


def test_all_field_film_slots_are_live() -> None:
    # No dead FiLM slot (the allocate-vs-consume off-by-one wasted slot 0). Perturbing each
    # slot must change the potential.
    torch.manual_seed(0)
    m = InputConvexNeuralNetwork(hidden_channels=16, n_layers=4, field_dim=1).eval()
    x = torch.rand(2, 1, 16, 16)
    fld = torch.full((2, 1), 3.0)
    base = m.potential(x, fld).detach().clone()
    n_slots = m.field_mlp.out_features // (2 * m.hidden)
    assert n_slots == m.n_layers - 2  # sized to what the loop consumes (no dead slot)
    for s in range(n_slots):
        m2 = InputConvexNeuralNetwork(hidden_channels=16, n_layers=4, field_dim=1).eval()
        m2.load_state_dict(m.state_dict())
        with torch.no_grad():
            m2.field_mlp.bias.view(m.n_layers - 2, 2, m.hidden)[s] += 1.0
        assert not torch.allclose(m2.potential(x, fld).detach(), base), f"slot {s} dead"


def test_rejects_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        InputConvexNeuralNetwork(in_channels=2)


# --- intensity_adaptation knob (Route 0/1/2): let the convex map adapt intensity/contrast ---
# Co-registered, normalised source/7T pairs leave a CONTRAST residual, not a geometric one. The
# default ("none") mean-pools the potential to a ~1/N-damped gradient => the map is pinned at
# identity. Route 1 ("undamped") un-damps it; Route 2 ("transfer") composes a field-conditioned
# monotone per-pixel intensity transfer. Both must stay monotone (Brenier-valid) under convexity.


def test_unknown_intensity_adaptation_raises() -> None:
    # Pitfall #9/#15: an unknown knob value must RAISE, never silently degrade to a default.
    with pytest.raises(ValueError, match="intensity_adaptation"):
        InputConvexNeuralNetwork(intensity_adaptation="bogus")


def _potential_grad_mean(mode: str) -> float:
    torch.manual_seed(0)
    m = InputConvexNeuralNetwork(
        hidden_channels=16, n_layers=3, intensity_adaptation=mode
    ).eval()
    x = torch.rand(2, 1, 32, 32).requires_grad_(True)
    (g,) = torch.autograd.grad(m.potential(x).sum(), x)
    return float(g.abs().mean())


def test_undamped_unblocks_the_potential_gradient() -> None:
    # THE mechanism: "none" mean-pools (grad ~ O(1/N), identity-pinned); "undamped" sum-pools so
    # the per-pixel intensity gradient is O(1). Same seeded weights, only the pooling differs.
    none_g, undamped_g = _potential_grad_mean("none"), _potential_grad_mean("undamped")
    assert undamped_g > 50.0 * none_g


def test_undamped_map_stays_monotone() -> None:
    # Sum-pool of a convex function is still convex => its gradient map stays monotone (Brenier).
    torch.manual_seed(0)
    m = InputConvexNeuralNetwork(
        hidden_channels=16, n_layers=3, intensity_adaptation="undamped"
    ).eval()
    a, b = torch.rand(6, 1, 16, 16), torch.rand(6, 1, 16, 16)
    ta, tb = m.brenier_map(a), m.brenier_map(b)
    mono = ((ta - tb) * (a - b)).flatten(1).sum(1).detach()
    assert float(mono.min()) >= -1e-3


def _fit_intensity(mode: str, *, field_dim: int = 0, slope: float = 1.0, shift: float = 0.2) -> float:
    # Fit a monotone affine intensity remap tgt = slope*x + shift. "none" is pinned at identity
    # (grad psi ~ O(1/N)) so it cannot reach any intensity offset; the adapting modes can.
    torch.manual_seed(0)
    x = torch.rand(4, 1, 20, 20)
    fld = torch.rand(4, 1) if field_dim else None
    tgt = slope * x + shift

    torch.manual_seed(0)
    m = InputConvexNeuralNetwork(
        hidden_channels=16, n_layers=3, field_dim=field_dim, intensity_adaptation=mode
    )
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    loss = None
    for _ in range(150):
        opt.zero_grad()
        loss = F.l1_loss(m.brenier_map(x, fld), tgt)
        loss.backward()
        opt.step()
    assert loss is not None
    return float(loss.detach())


def test_undamped_adapts_brightness_where_none_cannot() -> None:
    # A pure intensity offset (tgt = x + 0.2): the convex displacement map can realise it (its
    # affine input-skip contributes a constant gradient), "none" cannot (gradient ~ 0).
    assert _fit_intensity("undamped", slope=1.0, shift=0.2) < 0.5 * _fit_intensity(
        "none", slope=1.0, shift=0.2
    )


def test_undamped_convex_map_cannot_compress_contrast() -> None:
    # Documents WHY Route 2 exists: a convex displacement map has T'(x)=1+map_scale*psi'' >= 1, so
    # it can only EXPAND intensities, never COMPRESS them (slope < 1). The transfer route (learned
    # gain(b) < 1) lifts exactly this ceiling. Prevents "simplifying" Route 2 away.
    undamped_l1 = _fit_intensity("undamped", slope=0.4, shift=0.3)
    transfer_l1 = _fit_intensity("transfer", field_dim=1, slope=0.4, shift=0.3)
    assert transfer_l1 < 0.6 * undamped_l1


def test_transfer_map_is_monotone_in_input() -> None:
    # T(x) = g_b(x) + map_scale*grad psi(x): diag(g_b'(x)) (PSD, g_b monotone) + map_scale*Hess psi
    # (PSD, psi convex) => PSD => monotone. Perturb the transfer so it is non-trivial.
    torch.manual_seed(0)
    m = InputConvexNeuralNetwork(
        hidden_channels=16, n_layers=3, field_dim=1, intensity_adaptation="transfer"
    ).eval()
    with torch.no_grad():
        for p in m.transfer_mlp.parameters():
            p.normal_(0.0, 0.5)
    a, b = torch.rand(6, 1, 16, 16), torch.rand(6, 1, 16, 16)
    fld = torch.rand(6, 1)
    ta, tb = m.brenier_map(a, fld), m.brenier_map(b, fld)
    mono = ((ta - tb) * (a - b)).flatten(1).sum(1).detach()
    assert float(mono.min()) >= -1e-3


def test_transfer_is_field_conditioned() -> None:
    # The intensity transfer reads the SOURCE field: distinct fields => distinct transfer curves.
    torch.manual_seed(0)
    m = InputConvexNeuralNetwork(
        hidden_channels=16, n_layers=3, field_dim=1, intensity_adaptation="transfer"
    ).eval()
    with torch.no_grad():
        for p in m.transfer_mlp.parameters():
            p.normal_(0.0, 0.5)
    x = torch.rand(2, 1, 16, 16)
    lo = m.brenier_map(x, torch.tensor([[0.1], [0.1]]))
    hi = m.brenier_map(x, torch.tensor([[7.0], [7.0]]))
    assert not torch.allclose(lo.detach(), hi.detach())


def test_transfer_near_identity_at_init() -> None:
    # g_b init ~ identity (gain=1, bias=0, basis weights ~0) keeps the OT-sane near-identity start.
    torch.manual_seed(0)
    m = InputConvexNeuralNetwork(
        hidden_channels=16, n_layers=3, field_dim=1, intensity_adaptation="transfer"
    ).eval()
    x = torch.rand(4, 1, 16, 16)
    t = m.brenier_map(x, torch.rand(4, 1)).detach()
    assert float((t - x).norm() / x.norm()) < 0.5


def test_transfer_adapts_contrast() -> None:
    # Compressive remap (slope 0.5): only an intensity-adapting map fits it; "none" stays pinned.
    assert _fit_intensity("transfer", field_dim=1, slope=0.5, shift=0.25) < 0.6 * _fit_intensity(
        "none", field_dim=1, slope=0.5, shift=0.25
    )
