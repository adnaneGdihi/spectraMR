r"""Unit tests for the DL-BAE dispersion-latent Bloch autoencoder.

Targets ``spectramr.models.physics_ae.disp_bloch_ae``.

The load-bearing property is **identifiability**: one magnitude per field must
determine the :math:`2P+3` latent constants per voxel, so the arm needs
:math:`M \ge 2P+3` distinct fields (#2417). An under-determined arm must fail loudly at construction rather than
train to a rank-deficient optimum that looks converged (pitfall #9).

Also asserted: the latent carries no field-specific quantity, so the decoder can
render at a field never seen in training -- the property that separates DL-BAE
from a field-conditioned encoder.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.physics_ae.disp_bloch_ae import DispersionBlochAutoencoder

pytestmark = pytest.mark.unit

FIVE_FIELDS = (0.1, 1.5, 3.0, 5.0, 7.0)
SEVEN_FIELDS = (0.055, 0.1, 0.3, 1.5, 3.0, 5.0, 7.0)


def _ae(**kw) -> DispersionBlochAutoencoder:
    torch.manual_seed(0)
    kw.setdefault("fields_present", FIVE_FIELDS)
    kw.setdefault("hidden_channels", 8)
    kw.setdefault("depth", 2)
    n = len(kw["fields_present"])
    kw.setdefault("repetition_times_s", tuple(0.4 + 0.1 * m for m in range(n)))
    kw.setdefault("echo_times_s", tuple(0.010 + 0.002 * m for m in range(n)))
    return DispersionBlochAutoencoder(**kw)


@pytest.mark.parametrize(
    ("n_pools", "n_fields"),
    [(1, 5), (2, 7), (1, 7)],
)
def test_identifiable_configurations_construct(n_pools: int, n_fields: int) -> None:
    """M >= 2P+3 is satisfiable and must construct."""
    ae = _ae(fields_present=SEVEN_FIELDS[:n_fields], n_pools=n_pools)
    assert ae.n_pools == n_pools


@pytest.mark.parametrize(("n_pools", "n_fields"), [(1, 3), (1, 4), (2, 5), (2, 6), (3, 7)])
def test_under_determined_arm_raises(n_pools: int, n_fields: int) -> None:
    """M < 2P+3 is rank-deficient: raise, never train a meaningless latent.

    (2, 5) is the dlbae_brain_5field shape the old 2P+1 count passed (#2417).
    """
    with pytest.raises(ValueError, match="under-determined"):
        _ae(fields_present=SEVEN_FIELDS[:n_fields], n_pools=n_pools)


@pytest.mark.parametrize("missing", ["repetition_times_s", "echo_times_s"])
def test_missing_protocol_raises(missing: str) -> None:
    """No default TR/TE: a wrong protocol renders a plausible image silently (#1712)."""
    with pytest.raises(ValueError, match=missing):
        _ae(n_pools=1, **{missing: None})


def test_protocol_length_must_match_fields() -> None:
    with pytest.raises(ValueError, match="one positive time"):
        _ae(n_pools=1, repetition_times_s=(0.5, 0.5))


def test_each_channel_renders_at_its_own_protocol() -> None:
    """Channel m uses TR_m/TE_m, not one shared protocol (#1712)."""
    ae = _ae(n_pools=1)
    latent = ae.encode(torch.rand(1, 5, 4, 4))
    t1, t2 = ae.relaxation_maps(latent)
    tr = torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8]).view(1, -1, 1, 1)
    te = torch.tensor([0.010, 0.012, 0.014, 0.016, 0.018]).view(1, -1, 1, 1)
    expected = latent["rho"] * (1 - torch.exp(-tr / t1)) * torch.exp(-te / t2)
    torch.testing.assert_close(ae.decode(latent), expected)


def test_explicit_fields_need_an_explicit_protocol() -> None:
    ae = _ae(n_pools=1)
    latent = ae.encode(torch.rand(1, 5, 4, 4))
    with pytest.raises(ValueError, match="tr_s and te_s"):
        ae.decode(latent, torch.tensor([0.5]))


def test_duplicate_fields_do_not_count_toward_identifiability() -> None:
    """Repeating a field adds no rank -- it must not buy an extra pool."""
    with pytest.raises(ValueError, match="DISTINCT"):
        _ae(fields_present=(1.5, 1.5, 1.5, 1.5, 1.5), n_pools=1)


def test_non_positive_field_raises() -> None:
    with pytest.raises(ValueError, match="positive field strengths"):
        _ae(fields_present=(0.0, 1.5, 3.0, 5.0, 7.0), n_pools=1)


def test_inverted_tau_bounds_raise() -> None:
    with pytest.raises(ValueError, match="0 < min < max"):
        _ae(tau_c_bounds=(1e-6, 1e-11))


def test_encode_shapes_and_physiological_tau() -> None:
    """The latent is per-voxel, and tau_c is squashed into the declared bounds."""
    ae = _ae(fields_present=SEVEN_FIELDS, n_pools=2, tau_c_bounds=(1e-11, 1e-6))
    latent = ae.encode(torch.rand(2, 7, 8, 8))
    assert latent["rho"].shape == (2, 1, 8, 8)
    assert latent["b"].shape == (2, 2, 8, 8)
    assert latent["tau_c"].shape == (2, 2, 8, 8)
    assert torch.all(latent["tau_c"] >= 1e-11) and torch.all(latent["tau_c"] <= 1e-6)
    # Positivity heads: negative rates are unphysical.
    for key in ("rho", "a0", "c0", "b"):
        assert torch.all(latent[key] > 0.0), key


def test_wrong_channel_count_raises() -> None:
    """The stack must match fields_present -- silently reinterpreting it is worse."""
    ae = _ae(n_pools=1)
    with pytest.raises(ValueError, match="multi-field stack"):
        ae.encode(torch.rand(2, 3, 8, 8))


def test_latent_renders_at_an_unseen_field() -> None:
    """Field-invariance by construction: nothing in the latent names a field."""
    ae = _ae(n_pools=1)
    latent = ae.encode(torch.rand(1, 5, 6, 6))
    unseen = torch.tensor([0.5])  # not in FIVE_FIELDS
    out = ae.decode(latent, unseen, tr_s=0.5, te_s=0.015)
    assert out.shape == (1, 1, 6, 6)
    assert torch.isfinite(out).all()


def test_forward_roundtrip_shape_and_grad() -> None:
    ae = _ae(n_pools=1)
    x = torch.rand(2, 5, 8, 8)
    recon, _latent = ae(x)
    assert recon.shape == x.shape
    recon.sum().backward()
    assert any(p.grad is not None for p in ae.parameters())


def test_relaxation_maps_are_positive_and_finite() -> None:
    ae = _ae(n_pools=1)
    latent = ae.encode(torch.rand(1, 5, 6, 6))
    t1, t2 = ae.relaxation_maps(latent)
    assert t1.shape == (1, 5, 6, 6)
    assert torch.all(t1 > 0.0) and torch.all(t2 > 0.0)
    assert torch.isfinite(t1).all() and torch.isfinite(t2).all()
