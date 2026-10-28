"""R counts acquired phase-encode lines only (#2160), on non-square planes.

The supports below are M4Raw-like: a contiguous run of acquired columns with
never-acquired edges, 195/256 (T1/T2) and 198/256 (FLAIR) scaled onto the plane.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.diffusion.acquired_line_budget import (
    check_declared_lines,
    dynamic_lines,
)
from spectramr.models.diffusion.kspace_process import (
    KSpaceUndersamplingProcess,
    resolve_undersampling_kwargs,
)

LADDER = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def _process(axis: int, family: str = "density_nested", **over: object):
    arm = {
        "acceleration_type": family,
        "base_acceleration": 1.0,
        "max_acceleration": 32.0,
        "center_fraction": 0.08,
        "min_center_fraction": 0.02,
        "acceleration_range": LADDER,
        "mask_direction": "phase",
        "schedule_type": "step",
        "mask_seed": 42,
        "enforce_nested": family == "density_nested",
        **over,
    }
    kwargs = resolve_undersampling_kwargs(arm, phase_encode_axis=axis)
    return KSpaceUndersamplingProcess(num_timesteps=len(LADDER), **kwargs)


def _support(n: int, acquired: int, batch: int = 2) -> torch.Tensor:
    """Contiguous acquired lines with never-acquired edges, M4Raw's partial phase FOV."""
    lo = (n - acquired) // 2 - 2
    s = torch.zeros(batch, n, dtype=torch.bool)
    s[:, lo : lo + acquired] = True
    return s


def _lines(process, masks: torch.Tensor) -> torch.Tensor:
    across = -2 if process.line_dim == -1 else -1
    return (masks[:, 0] > 0).any(dim=across)


@pytest.mark.parametrize(("h", "w"), [(48, 80), (40, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("family", ["density_nested", "equispaced", "random"])
def test_a_full_support_reproduces_the_cascade_bit_for_bit(h, w, axis, family) -> None:
    """Keeps every pinned mask and ladder test valid; the ladder gate builds without data."""
    process = _process(axis, family)
    n = w if axis == -1 else h
    full = torch.ones(2, n, dtype=torch.bool)
    for t in range(len(LADDER)):
        steps = torch.full((2,), t, dtype=torch.long)
        before = process.cascade_batch_masks(steps, (h, w), torch.device("cpu"))
        after = process.cascade_batch_masks(steps, (h, w), torch.device("cpu"), acquired=full)
        assert torch.equal(before, after)


@pytest.mark.parametrize(("h", "w"), [(48, 80), (40, 64)])
def test_partial_support_counts_r_over_acquired_lines_and_stays_nested(h, w) -> None:
    process = _process(-1)
    support = _support(w, round(w * 195 / 256))
    previous = None
    for t, declared in enumerate(LADDER):
        steps = torch.full((2,), t, dtype=torch.long)
        lines = _lines(process, process.cascade_batch_masks(steps, (h, w), "cpu", support))
        assert not bool((lines & ~support).any()), "a never-acquired line was selected"
        assert bool(lines[:, process._line_ranking((h, w)).core].all()), "ACS core lost"
        realised = support.sum(-1).float() / lines.sum(-1).float()
        budget = round(int(process._line_ranking((h, w)).kept[t]) * int(support[0].sum()) / w)
        assert int(lines[0].sum()) == budget
        assert torch.allclose(realised, torch.full_like(realised, declared), rtol=0.35)
        if previous is not None:
            assert not bool((lines & ~previous).any()), f"t={t} re-adds a line"
        previous = lines


def test_a_mask_that_selects_a_never_acquired_line_is_caught() -> None:
    """PLANTED: the pre-#2160 mask ranks all lines, so it spends budget on empty ones."""
    process = _process(-1)
    support = _support(80, 61)
    steps = torch.full((2,), 1, dtype=torch.long)
    old = _lines(process, process.cascade_batch_masks(steps, (48, 80), "cpu"))
    new = _lines(process, process.cascade_batch_masks(steps, (48, 80), "cpu", support))
    assert bool((old & ~support).any())
    assert not bool((new & ~support).any())


def test_an_acs_core_wider_than_the_top_budget_raises() -> None:
    """PLANTED: a 10-line core against a 32x budget of round(80 * 40/80 / 32) = 1 line."""
    process = _process(-1, min_center_fraction=0.125, center_fraction=0.125)
    steps = torch.full((2,), len(LADDER) - 1, dtype=torch.long)
    with pytest.raises(ValueError, match="min_center_fraction"):
        process.cascade_batch_masks(steps, (48, 80), "cpu", _support(80, 40))


def test_q_sample_reads_the_support_off_the_data() -> None:
    process = _process(-1)
    x = torch.randn(2, 4, 48, 80)
    support = _support(80, 61)
    x = x * support[:, None, None, :]
    process.eval()
    _, masks = process.q_sample(x, torch.tensor([4, 4]))
    lines = _lines(process, masks)
    assert not bool((lines & ~support).any())
    assert int(lines[0].sum()) == round(int(process._line_ranking((48, 80)).kept[4]) * 61 / 80)


def test_dynamic_draws_rank_only_acquired_lines() -> None:
    n = 80
    support = _support(n, 61)[0]
    core = torch.zeros(n, dtype=torch.bool)
    core[38:42] = True
    draw = core.clone()
    draw[[3, 10, 50, 70, 79]] = True
    lines = dynamic_lines(draw, support, core, seed=7)
    assert not bool((lines & ~support).any())
    assert bool(lines[core].all())
    assert int(lines.sum()) == round(int(draw.sum()) * 61 / n)
    assert torch.equal(dynamic_lines(draw, torch.ones(n, dtype=torch.bool), core, seed=7), draw)


def test_dynamic_training_masks_respect_the_support() -> None:
    process = _process(-1, enable_dynamic_mask=True)
    x = torch.randn(2, 4, 48, 80) * _support(80, 61)[:, None, None, :]
    process.train(True)
    _, masks = process.q_sample(x, torch.tensor([3, 3]))
    assert not bool((_lines(process, masks) & ~_support(80, 61)).any())


class TestDeclaredCounts:
    def test_a_matching_per_contrast_mapping_passes(self) -> None:
        acquired = torch.cat([_support(256, 195, 1), _support(256, 198, 1)])
        out = check_declared_lines({"T1": 195, "FLAIR": 198}, acquired, ["T1", "FLAIR"])
        assert out["measured"] == [195, 198]

    def test_a_disagreeing_count_raises(self) -> None:
        """PLANTED: a manifest whose files acquire a different count than declared."""
        acquired = _support(256, 195, 2)
        with pytest.raises(ValueError, match="FLAIR: declared 198, measured 195"):
            check_declared_lines({"FLAIR": 198, "T1": 195}, acquired, ["FLAIR", "T1"])
        with pytest.raises(ValueError, match="declared 198"):
            check_declared_lines(198, acquired, None)

    def test_rows_flattened_from_depth_take_their_subject_contrast(self) -> None:
        acquired = torch.cat([_support(256, 195, 3), _support(256, 198, 3)])
        check_declared_lines({"T1": 195, "FLAIR": 198}, acquired, ["T1", "FLAIR"])

    @pytest.mark.parametrize("contrasts", [None, ["GRE", "T1"]])
    def test_a_mapping_without_a_name_for_every_row_raises(self, contrasts) -> None:
        with pytest.raises(ValueError, match="contrast"):
            check_declared_lines({"T1": 195}, _support(256, 195, 2), contrasts)

    def test_measured_records_without_checking(self) -> None:
        acquired = torch.cat([_support(320, 224, 1), _support(320, 192, 1)])
        assert check_declared_lines("measured", acquired, None)["measured"] == [192, 224]


def test_the_ranking_alone_reproduces_a_nested_cascade() -> None:
    """The mechanism, not the full-support shortcut: top ``kept_t`` by rank == the cascade."""
    from spectramr.models.diffusion.acquired_line_budget import budget_lines

    process = _process(-1)
    ranking = process._line_ranking((48, 80))
    full = torch.ones(1, 80, dtype=torch.bool)
    for t in range(len(LADDER)):
        expected = process.mask_at(t, (48, 80)).any(dim=0)
        assert torch.equal(budget_lines(ranking, torch.tensor([t]), full)[0], expected)


@pytest.mark.parametrize("mode", ["replace_freeze_dc", "additive_normalized", "support_ordered_dc"])
def test_every_reverse_rung_is_built_over_the_measured_lines(mode) -> None:
    """The reverse loop's rungs must come from the support the measurement was built on."""
    from tests.unit.models.diffusion import _reverse_phantom as P

    process = _process(-1, family="density_nested")
    target = P.kspace()
    support = _support(P.W, 50)
    seen: list[object] = []
    original = process.cascade_batch_masks

    def spy(*args, **kwargs):
        seen.append(kwargs.get("acquired"))
        return original(*args, **kwargs)

    process.cascade_batch_masks = spy
    head = torch.full((2,), len(LADDER) - 1, dtype=torch.long)
    mask = original(head, (P.H, P.W), "cpu", acquired=support)
    sampler = P.sampler(
        P.Oracle(target, process), mode, num_timesteps=len(LADDER), sampling_steps=len(LADDER)
    )
    sampler.sample(target * mask, mask, measured_lines=support)
    assert seen and all(a is support for a in seen)


def test_the_registered_cold_mri_wrapper_carries_the_measured_lines() -> None:
    """The sampler validation actually gets (``get_sampler("cold_mri")``) must accept the lines.

    The sibling test above builds the core class directly, which is how the wrapper's missing
    parameter stayed invisible: the generator refuses a sampler that does not name
    ``measured_lines``, so every line-structured arm crashed its first validation.
    """
    from spectramr.models.diffusion.samplers import get_sampler
    from tests.unit.models.diffusion import _reverse_phantom as P

    process = _process(-1, family="density_nested")
    target = P.kspace()
    support = _support(P.W, 50)
    seen: list[object] = []
    original = process.cascade_batch_masks

    def spy(*args, **kwargs):
        seen.append(kwargs.get("acquired"))
        return original(*args, **kwargs)

    process.cascade_batch_masks = spy
    head = torch.full((2,), len(LADDER) - 1, dtype=torch.long)
    mask = original(head, (P.H, P.W), "cpu", acquired=support)
    model = P.Oracle(target, process)
    sampler = get_sampler(
        "cold_mri",
        model=model,
        num_timesteps=len(LADDER),
        max_acceleration=process.max_accel,
        center_fraction=0.08,
        sampling_steps=len(LADDER),
        reverse_mode="replace_freeze_dc",
        kspace_log_scaled=False,
    )
    sampler.sample(target * mask, mask, measured_lines=support)
    assert seen and all(a is support for a in seen)


@pytest.mark.parametrize("mode", ["replace_freeze_dc", "support_ordered_dc"])
def test_the_terminal_step_leaves_never_acquired_lines_zero(mode) -> None:
    """The model writes content on the padded lines; the reconstruction must not keep it.

    Those lines are exact zeros in the target, so a written value there is scored as
    error by every whole-plane metric. The integrity check (on) must also count them
    as owned by the measurement rather than as never written.
    """
    from tests.unit.models.diffusion import _reverse_phantom as P

    process = _process(-1, family="density_nested")
    hallucinating = P.kspace()  # content on every column, the padded ones included
    support = _support(P.W, 50)
    plane = support[:, None, None, :].float()
    head = torch.full((2,), len(LADDER) - 1, dtype=torch.long)
    mask = process.cascade_batch_masks(head, (P.H, P.W), "cpu", acquired=support)
    sampler = P.sampler(
        P.Oracle(hallucinating, process),
        mode,
        num_timesteps=len(LADDER),
        sampling_steps=len(LADDER),
        integrity_check="raise",
    )
    out = sampler.sample(hallucinating * plane * mask, mask, measured_lines=support)
    assert torch.equal(out * (1 - plane), torch.zeros_like(out))
    assert bool((out * plane).abs().sum() > 0)


def test_planted_without_measured_lines_the_terminal_step_fills_the_padding() -> None:
    from tests.unit.models.diffusion import _reverse_phantom as P

    process = _process(-1, family="density_nested")
    hallucinating = P.kspace()
    support = _support(P.W, 50)
    plane = support[:, None, None, :].float()
    head = torch.full((2,), len(LADDER) - 1, dtype=torch.long)
    mask = process.cascade_batch_masks(head, (P.H, P.W), "cpu", acquired=support)
    sampler = P.sampler(
        P.Oracle(hallucinating, process),
        "replace_freeze_dc",
        num_timesteps=len(LADDER),
        sampling_steps=len(LADDER),
    )
    out = sampler.sample(hallucinating * plane * mask, mask)
    assert bool((out * (1 - plane)).abs().sum() > 0)
