"""Support-ordered commitment (Algorithm D): lines are committed outward from the measured ones."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.diffusion.reverse_support_ordered import commit_radius, line_distance
from tests.unit.models.diffusion import _reverse_phantom as P

pytestmark = pytest.mark.unit

RADIUS = 3


class _CallIndex(torch.nn.Module):
    """Writes ``scale * (call + 1)`` everywhere, so a committed bin records the step that froze it."""

    def __init__(self, proc, scale: float) -> None:
        super().__init__()
        self.kspace_process = proc
        self.scale = scale
        self.calls = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_kw: object) -> torch.Tensor:
        self.calls += 1
        return torch.full_like(x, self.scale * self.calls)


@pytest.fixture(scope="module")
def setting():
    proc = P.process(8.0)
    target = P.kspace()
    mask = P.mask(proc)
    return proc, target, target * mask, mask


def test_line_distance_counts_rows_to_the_nearest_measured_row() -> None:
    mask = torch.zeros(1, 1, 8, 4)
    mask[..., [2, 5], :] = 1.0
    assert line_distance(mask, torch.zeros(1, 2, 8, 4)).tolist() == [[2, 1, 0, 1, 1, 0, 1, 2]]


def test_line_distance_counts_columns_on_a_column_mask() -> None:
    """On a mask that removes columns (#2160) the lines are columns, 6x10 here."""
    mask = torch.zeros(1, 1, 6, 10)
    mask[..., [3, 7]] = 1.0
    assert line_distance(mask, torch.zeros(1, 2, 6, 10), -1).tolist() == [
        [3, 2, 1, 0, 1, 2, 1, 0, 1, 2]
    ]
    # Read as rows, every row of a column mask is touched: the defect shape.
    assert line_distance(mask, torch.zeros(1, 2, 6, 10)).tolist() == [[0] * 6]


def test_the_commit_radius_grows_from_one_to_the_declared_radius() -> None:
    radii = [commit_radius(i, 8, RADIUS) for i in range(7)]
    assert radii[0] == 1 and radii[-1] == RADIUS
    assert radii == sorted(radii)
    assert commit_radius(0, 2, RADIUS) == 1


def test_the_oracle_is_reproduced_and_every_bin_is_written(setting) -> None:
    proc, target, meas, mask = setting
    sampler = P.sampler(
        P.Oracle(target, proc),
        "support_ordered_dc",
        support_radius=RADIUS,
        reverse_clip_ratio=100.0,
        integrity_check="raise",
    )
    out = sampler.sample(meas, mask, start_timestep=P.HEAD)
    assert torch.allclose(out, target, atol=1e-5)
    assert sampler.reverse_stats["integrity_steps_verified"] == sampler.last_effective_steps > 0


def test_lines_are_committed_outward_from_the_measured_ones(setting) -> None:
    """The reveal direction: a line is never frozen before a line nearer to the data."""
    proc, _, meas, mask = setting
    scale = 1e-4 * float(meas.abs().max())
    model = _CallIndex(proc, scale)
    sampler = P.sampler(model, "support_ordered_dc", support_radius=RADIUS)
    out = sampler.sample(meas, mask, start_timestep=P.HEAD)

    distance = line_distance(mask, meas)[0]
    step = (out[0, 0, :, 0] / scale).round()
    free = distance > 0
    order = sorted(zip(distance[free].tolist(), step[free].tolist(), strict=True))
    assert [s for _, s in order] == sorted(s for _, s in order)
    for d in range(1, RADIUS + 1):
        first = 1 + min(
            i for i in range(model.calls - 1) if commit_radius(i, model.calls, RADIUS) >= d
        )
        assert set(step[distance == d].tolist()) == {float(first)}, d
    assert set(step[distance > RADIUS].tolist()) == {float(model.calls)}


def test_hard_dc_keeps_an_outer_band_above_the_envelope_pinned() -> None:
    """PLANTED (#2310): the ceiling clamp ran after hard DC on the whole plane."""
    from tests.unit.models.diffusion import _reverse_phantom as phantom

    out, meas, acquired = phantom.band_local_outlier_run("support_ordered_dc")
    assert torch.equal(out * acquired, meas)


def test_train_mode_reconstructs_what_eval_mode_does() -> None:
    """PLANTED (#2471): the model's input support came from ``q_sample``, a dynamic training mask
    in train mode, so both the input the model saw and the reconstruction changed with the mode."""
    from tests.unit.models.diffusion import _reverse_phantom as phantom

    outs, inputs = phantom.module_mode_runs("support_ordered_dc")
    for run in inputs[1:]:
        assert len(run) == len(inputs[0])
        assert all(torch.equal(a, b) for a, b in zip(run, inputs[0], strict=True))
    assert all(torch.equal(out, outs[0]) for out in outs[1:])
