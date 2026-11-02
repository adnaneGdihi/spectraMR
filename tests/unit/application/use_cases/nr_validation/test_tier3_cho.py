"""Tests for §8.3 Tier 3 — CHO task concordance scorer.

Validates the public ``score_cho_concordance`` contract (never raises, returns a
``TierResult`` per requested metric, finite-or-NaN scores, finite mean d'), the
physics guarantee that CHO detectability falls as noise rises, and the
``_insert_blob`` SKE-signal helper.
"""

from __future__ import annotations

import math

import torch

from spectramr.application.use_cases.nr_validation.cards import TierResult
from spectramr.application.use_cases.nr_validation.tier3_cho import (
    _ensemble,
    _insert_blob,
    score_cho_concordance,
)
from spectramr.infrastructure.physics.model_observer import (
    ChannelizedHotellingObserver,
    gabor_channel_bank,
)


def _smooth_ref(h: int = 16, w: int = 16, seed: int = 0) -> torch.Tensor:
    """A small, deterministic, smooth gradient reference ``[1, H, W]``."""
    torch.manual_seed(seed)
    ys = torch.linspace(0.2, 0.9, h).view(h, 1)
    xs = torch.linspace(0.1, 0.8, w).view(1, w)
    grad = (ys + xs) / 2.0
    # A touch of fixed low-amplitude structure so dynamic range is non-trivial.
    g = torch.Generator().manual_seed(seed)
    grad = grad + 0.02 * torch.rand((h, w), generator=g)
    return grad.unsqueeze(0)


def test_insert_blob_raises_central_pixel() -> None:
    """The blob must lift the central pixel above the local background."""
    img = torch.zeros((1, 16, 16))
    out = _insert_blob(img, contrast=0.25, radius_frac=0.06)
    h, w = 16, 16
    cy, cx = (h - 1) // 2, (w - 1) // 2
    center_val = float(out[0, cy, cx].item())
    corner_val = float(out[0, 0, 0].item())
    assert center_val > corner_val
    assert center_val > 0.0
    # Same shape / dtype preserved.
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_insert_blob_uses_dynamic_range() -> None:
    """Blob peak scales with the image dynamic range (contrast fraction).

    The geometric centre of a 16-px image is at index 7.5, so no pixel sits
    exactly on the Gaussian peak; the added amplitude at the nearest pixel is
    bounded above by ``contrast * dynamic_range`` and grows with ``contrast``.
    """
    base = _smooth_ref()
    dyn = float(base.amax().item() - base.amin().item())
    h, w = 16, 16
    cy, cx = (h - 1) // 2, (w - 1) // 2

    out_lo = _insert_blob(base, contrast=0.25, radius_frac=0.06)
    out_hi = _insert_blob(base, contrast=0.5, radius_frac=0.06)
    added_lo = float(out_lo[0, cy, cx].item() - base[0, cy, cx].item())
    added_hi = float(out_hi[0, cy, cx].item() - base[0, cy, cx].item())

    # Positive, scales with contrast, and never exceeds the analytic peak.
    assert added_lo > 0.0
    assert added_hi > added_lo
    assert added_lo <= 0.25 * dyn + 1e-6
    # Doubling contrast doubles the added amplitude (linear in contrast).
    assert abs(added_hi - 2.0 * added_lo) < 1e-5


def test_dprime_decreases_with_noise() -> None:
    """Physics guarantee: higher noise -> lower CHO detectability d'.

    Exercises the d' machinery directly (the same path the scorer uses) at two
    noise levels and asserts the harder task has the smaller d'.
    """
    ref = _smooth_ref()
    present_ref = _insert_blob(ref, contrast=0.4, radius_frac=0.08)
    absent_ref = ref
    h, w = 16, 16
    bank = gabor_channel_bank((h, w))

    def dprime(theta: float) -> float:
        present = _ensemble(
            present_ref, "gaussian_noise", theta, base_seed=11, n_ensemble=24
        )
        absent = _ensemble(
            absent_ref, "gaussian_noise", theta, base_seed=22, n_ensemble=24
        )
        obs = ChannelizedHotellingObserver(bank)
        d_sq = obs.detectability(present, absent)
        assert math.isfinite(d_sq)
        return math.sqrt(max(0.0, d_sq))

    d_low = dprime(0.1)
    d_high = dprime(0.9)
    assert d_low > d_high  # detectability falls as noise rises
    assert d_high >= 0.0


def test_score_cho_concordance_contract() -> None:
    """Returns a TierResult per requested metric; never raises; finite mean d'."""
    refs = [("ref0", _smooth_ref(seed=0)), ("ref1", _smooth_ref(seed=1))]
    nr_names = ["psnr", "ssim"]
    out = score_cho_concordance(
        refs,
        nr_names,
        families=["gaussian_noise"],  # cheap single family
        n_severities=4,
        n_ensemble=12,
        seed=7,
    )
    assert set(out.keys()) == set(nr_names)
    for name in nr_names:
        res = out[name]
        assert isinstance(res, TierResult)
        assert res.tier == "cho_concordance"
        # score is finite or NaN, never an exception.
        assert isinstance(res.score, float)
        assert math.isfinite(res.score) or math.isnan(res.score)
        assert isinstance(res.passed, bool)
        # mean_dprime is finite for a real noise sweep.
        assert math.isfinite(res.detail["mean_dprime"])
        assert res.detail["n_severities"] == 4
        assert "per_family" in res.detail


def test_score_cho_concordance_empty_inputs_no_crash() -> None:
    """Degenerate inputs return failing TierResults rather than raising."""
    # No clean volumes.
    out = score_cho_concordance([], ["psnr"], families=["gaussian_noise"])
    assert out["psnr"].passed is False
    assert math.isnan(out["psnr"].score)

    # Unknown family only -> no known families.
    refs = [("r", _smooth_ref())]
    out2 = score_cho_concordance(refs, ["psnr"], families=["does_not_exist"])
    assert out2["psnr"].passed is False
    assert math.isnan(out2["psnr"].score)


def test_same_prefix_contents_get_different_tier3_seeds(monkeypatch) -> None:
    """Tier 3 shares the simulator's blake2b content seed, so ``phantom_0`` and
    ``phantom_1`` (identical first 8 bytes) no longer draw one ensemble."""
    from spectramr.application.use_cases.nr_validation import tier3_cho

    seen: list[int] = []

    def _record(*_a, base_seed, **_k):
        seen.append(base_seed)
        raise RuntimeError("seed captured")

    monkeypatch.setattr(tier3_cho, "_ensemble", _record)
    monkeypatch.setattr(tier3_cho, "_mean_image", _record)
    seeds = []
    for cid in ("phantom_0", "phantom_1"):
        seen.clear()
        score_cho_concordance([(cid, _smooth_ref())], ["psnr"], n_severities=1, seed=3)
        seeds.append(seen[0])
    assert seeds[0] != seeds[1]
