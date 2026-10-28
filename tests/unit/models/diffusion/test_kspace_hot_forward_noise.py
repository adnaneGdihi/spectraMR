"""Training-time ("hot") forward noise on the unmeasured bins of the k-space process.

``model.model_kwargs.forward_noise_sigma`` adds ``sigma_t * RMS(measured x_t) * eps`` on
the bins ``M_t`` does not acquire, with ``sigma_t`` from ``forward_noise_schedule``. The
reverse loop re-noises the same bins before every denoiser call. At sigma 0 nothing of
this runs: no draw, and bit-identical outputs.
"""

from __future__ import annotations

import copy

import pytest
import torch

from spectramr.models.diffusion.kspace_process import (
    FORWARD_NOISE_SCHEDULES,
    VALID_REVERSE_MODES,
    KSpaceUndersamplingProcess,
    PhysicsInformedColdDiffusion,
    validate_forward_noise,
    validate_forward_noise_exclusive,
)
from tests.unit.models.diffusion.test_reverse_ladder_contract import (
    LADDERS,
    SIZE,
    _build,
    _target,
    _timesteps,
)

T = 10
MODES = sorted(VALID_REVERSE_MODES)


def _process(sigma: float = 0.0, schedule: str = "linear_t") -> KSpaceUndersamplingProcess:
    return KSpaceUndersamplingProcess(
        num_timesteps=T,
        max_acceleration=4.0,
        center_fraction=0.1,
        seed=0,
        train_identity_rung=True,
        forward_noise_sigma=sigma,
        forward_noise_schedule=schedule,
    )


def _rms_over(x: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
    """Per-sample complex RMS of ``x`` over the measured bins (real-stacked: channel pairs)."""
    power = x.abs() ** 2 if torch.is_complex(x) else 2.0 * x * x
    m = measured.float().expand_as(power)
    return ((power * m).flatten(1).sum(1) / m.flatten(1).sum(1)).sqrt()


class _Recorder(torch.nn.Module):
    """A denoiser stand-in that records every call's timestep, input and mask keyword."""

    def __init__(self, process, kind: str, target: torch.Tensor | None = None) -> None:
        super().__init__()
        self.kspace_process = process
        self.kind = kind
        self.target = target
        self.calls: list[tuple[int, torch.Tensor, torch.Tensor | None]] = []

    def forward(self, x, t, **kw):
        self.calls.append((int(t.reshape(-1)[0]), x.detach().clone(), kw.get("mask")))
        if self.kind == "oracle":
            return self.target.clone()
        return x.clone()


# ── sigma = 0 is the old process ─────────────────────────────────────────────


def test_sigma_zero_q_sample_draws_no_rng_and_degrades_by_the_mask_only():
    process = _process()
    x = torch.randn(2, 2, SIZE, SIZE, dtype=torch.complex64)
    torch.manual_seed(4)
    state = torch.get_rng_state()
    x_t, masks = process.q_sample(x, torch.tensor([T - 1, 3]))
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(x_t, x * masks.expand_as(x))


def test_sigma_zero_sample_draws_no_global_rng():
    process = _process()
    model = _Recorder(process, "identity")
    sampler = PhysicsInformedColdDiffusion(
        model=model,
        num_timesteps=T,
        max_acceleration=4.0,
        center_fraction=0.1,
        dc_method="hard",
        reverse_mode="replace_freeze",
        kspace_log_scaled=False,
        sampler_seed=2,
    )
    mask = process.mask_at(T - 1, (SIZE, SIZE)).float().reshape(1, 1, SIZE, SIZE)
    measurement = torch.randn(1, 2, SIZE, SIZE) * mask
    torch.manual_seed(9)
    state = torch.get_rng_state()
    sampler.sample(measurement, mask, start_timestep=T - 1)
    assert torch.equal(torch.get_rng_state(), state)


# ── forward process ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("complex_layout", [True, False], ids=["complex", "real_stacked"])
def test_measured_bins_are_bit_identical_under_forward_noise(complex_layout):
    process = _process(sigma=0.4)
    shape = (2, 1, SIZE, SIZE) if complex_layout else (2, 2, SIZE, SIZE)
    dtype = torch.complex64 if complex_layout else torch.float32
    x = torch.randn(shape, dtype=dtype)
    x_t, masks = process.q_sample(x, torch.tensor([T - 1, 6]))
    measured = masks.expand_as(x).bool()
    assert torch.equal(x_t[measured], (x * masks.expand_as(x))[measured])
    assert not torch.equal(x_t[~measured], (x * masks.expand_as(x))[~measured])


def test_forward_noise_at_t0_is_zero():
    process = _process(sigma=0.8)
    x = torch.randn(1, 1, SIZE, SIZE, dtype=torch.complex64)
    x_t, masks = process.q_sample(x, torch.tensor([0]))
    assert torch.equal(x_t, x * masks.expand_as(x))


@pytest.mark.parametrize("schedule", sorted(FORWARD_NOISE_SCHEDULES))
@pytest.mark.parametrize("t_idx", [T - 1, T // 2])
@pytest.mark.parametrize("complex_layout", [True, False], ids=["complex", "real_stacked"])
def test_unmeasured_noise_std_is_sigma_t_times_measured_rms(schedule, t_idx, complex_layout):
    sigma = 0.35
    process = _process(sigma=sigma, schedule=schedule)
    torch.manual_seed(13)
    shape = (4, 1, SIZE, SIZE) if complex_layout else (4, 2, SIZE, SIZE)
    dtype = torch.complex64 if complex_layout else torch.float32
    x = torch.randn(shape, dtype=dtype) * 3.0
    t = torch.full((4,), t_idx, dtype=torch.long)
    x_t, masks = process.q_sample(x, t)
    clean = x * masks.expand_as(x)
    noise = (x_t - clean)[~masks.expand_as(x).bool()]

    fraction = float(FORWARD_NOISE_SCHEDULES[schedule](torch.tensor(float(t_idx)), float(T - 1)))
    expected_power = ((sigma * fraction) ** 2 * _rms_over(clean, masks) ** 2).mean()
    power = noise.abs().pow(2).mean() if complex_layout else noise.pow(2).mean() * 2.0
    assert float(power) == pytest.approx(float(expected_power), rel=0.06)


def test_passed_noise_is_used_verbatim():
    process = _process(sigma=0.5, schedule="linear_t")
    x = torch.randn(1, 2, SIZE, SIZE)
    t = torch.tensor([T - 1])
    x_t, masks = process.q_sample(x, t, noise=torch.ones_like(x))
    clean = x * masks.expand_as(x)
    level = 0.5 * (T - 1) / (T - 1)
    scale = level * _rms_over(clean, masks)
    expected = clean + (scale.view(-1, 1, 1, 1) / (2.0**0.5)) * (1.0 - masks.float())
    torch.testing.assert_close(x_t, expected)


# ── reverse process ──────────────────────────────────────────────────────────


def _reverse(process, mode: str, *, seed: int | None, kind: str = "oracle"):
    target = torch.randn(1, 2, SIZE, SIZE, generator=torch.Generator().manual_seed(3))
    model = _Recorder(process, kind, target)
    sampler = PhysicsInformedColdDiffusion(
        model=model,
        num_timesteps=T,
        max_acceleration=process.max_accel,
        center_fraction=process.center_fraction,
        dc_method="hard",
        reverse_mode=mode,
        reverse_clip_ratio=1e6,
        refine_alpha=0.5 if mode == "reveal_refine_dc" else None,
        support_radius=2 if mode == "support_ordered_dc" else None,
        kspace_log_scaled=False,
        sampler_seed=seed,
    )
    mask = process.mask_at(T - 1, (SIZE, SIZE)).float().reshape(1, 1, SIZE, SIZE)
    out = sampler.sample(target * mask, mask, start_timestep=T - 1)
    return out, model


@pytest.mark.parametrize("mode", MODES)
def test_seeded_reverse_reproduces_exactly_and_follows_the_seed(mode):
    process = _process(sigma=0.3)
    torch.manual_seed(21)
    state = torch.get_rng_state()
    # An identity denoiser: an oracle ignores its input, so no noise would reach the output.
    a, _ = _reverse(process, mode, seed=11, kind="identity")
    assert torch.equal(torch.get_rng_state(), state), "reverse noise must not touch global RNG"
    b, _ = _reverse(process, mode, seed=11, kind="identity")
    c, _ = _reverse(process, mode, seed=12, kind="identity")
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("ladder", list(LADDERS))
@pytest.mark.parametrize("mode", MODES)
def test_reverse_spy_noises_only_unmeasured_bins_and_keeps_the_line_count(ladder, mode):
    """The first call sees the measurement; noise lands only where the support is not.

    The measured-line count is read from the ``mask`` keyword the denoiser is handed, which
    is built from the support, never from the (noised) input. Counting nonzeros of the
    input would read every noised line as acquired.
    """
    doc, shared = _build(ladder)
    # A copy: the ladder builder memoises its process, and the other ladder tests read it at sigma 0.
    process = copy.deepcopy(shared)
    process.forward_noise_sigma, process.forward_noise_schedule = validate_forward_noise(
        0.3, "linear_t"
    )
    target = _target()
    model = _Recorder(process, "oracle", target)
    sampler = PhysicsInformedColdDiffusion(
        model=model,
        num_timesteps=_timesteps(doc),
        max_acceleration=process.max_accel,
        center_fraction=process.center_fraction,
        dc_method="hard",
        reverse_mode=mode,
        reverse_clip_ratio=1e6,
        refine_alpha=0.5 if mode == "reveal_refine_dc" else None,
        support_radius=2 if mode == "support_ordered_dc" else None,
        kspace_log_scaled=False,
        sampler_seed=4,
    )
    head = _timesteps(doc) - 1
    mask = process.mask_at(head, (SIZE, SIZE)).float().reshape(1, 1, SIZE, SIZE)
    sampler.sample(target * mask, mask, start_timestep=head)
    assert model.calls, "the sampler made no call"

    label, first_x, first_mask = model.calls[0]
    known = first_mask.expand_as(first_x) > 0
    measured = target * mask
    assert torch.equal(first_x[known], measured.expand_as(first_x)[known])
    assert (first_x[~known] != 0).any(), "unmeasured bins were not re-noised"

    ladder_rows = {row[0]: row for row in process.describe_ladder((SIZE, SIZE))}
    for label, _, call_mask in model.calls:
        lines = _line_count_from_mask(process, call_mask)
        expected = SIZE / ladder_rows[label][2]
        assert abs(lines - expected) <= 1.0, (
            f"t={label}: {lines} measured lines, expected {expected:.2f}"
        )


def _line_count_from_mask(process, mask: torch.Tensor) -> int:
    plane = mask.reshape(mask.shape[-2], mask.shape[-1]) > 0
    return int(plane.any(dim=1).sum()) if process.line_dim == -2 else int(plane.any(dim=0).sum())


# ── refusals ─────────────────────────────────────────────────────────────────


def test_forward_noise_with_sampler_sigma_is_refused_at_the_sampler():
    process = _process(sigma=0.2)
    with pytest.raises(ValueError, match="one"):
        PhysicsInformedColdDiffusion(
            model=_Recorder(process, "identity"),
            num_timesteps=T,
            max_acceleration=4.0,
            center_fraction=0.1,
            reverse_mode="replace_freeze",
            kspace_log_scaled=False,
            sampler_sigma=0.1,
        )


def test_exclusivity_helper_allows_one_owner_only():
    validate_forward_noise_exclusive(0.2, 0.0)
    validate_forward_noise_exclusive(0.0, 0.2)
    with pytest.raises(ValueError):
        validate_forward_noise_exclusive(0.2, 0.2)


def test_unknown_schedule_raises():
    with pytest.raises(ValueError, match="forward_noise_schedule"):
        validate_forward_noise(0.1, "cosine_t")
    with pytest.raises(ValueError, match="forward_noise_schedule"):
        _process(sigma=0.1, schedule="cosine_t")


@pytest.mark.parametrize("sigma", [-0.1, float("nan"), float("inf")])
def test_invalid_sigma_raises(sigma):
    with pytest.raises(ValueError, match="forward_noise_sigma"):
        validate_forward_noise(sigma, "linear_t")
