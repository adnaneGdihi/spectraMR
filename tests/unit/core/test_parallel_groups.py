"""The data-parallel layout read side: the global answer when no layout is active."""

from __future__ import annotations

import pytest

from spectramr.core import parallel_groups as pg

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _no_layout():
    pg.reset_dp_layout()
    yield
    pg.reset_dp_layout()


def test_without_a_layout_every_helper_is_the_global_answer() -> None:
    # Single process: no process group, so rank 0 of world 1.
    assert pg.current_dp_layout() is None
    assert pg.dp_rank() == 0
    assert pg.dp_world_size() == 1
    assert pg.dp_group() is None
    assert pg.dp_src_rank() == 0
    assert pg.is_dp_writer() is True


def test_an_active_layout_answers_every_helper() -> None:
    layout = pg.DPLayout(
        rank=1,
        world_size=2,
        group="dp-group-handle",
        src_rank=1,
        writer=False,
        dim_names=("pp", "dp_replicate", "dp_shard", "cp", "sp", "tp"),
        shape=(1, 2, 1, 1, 1, 2),
    )
    pg.activate_dp_layout(layout)
    assert pg.current_dp_layout() is layout
    assert pg.dp_rank() == 1
    assert pg.dp_world_size() == 2
    assert pg.dp_group() == "dp-group-handle"
    assert pg.dp_src_rank() == 1
    assert pg.is_dp_writer() is False


def test_reset_restores_the_global_answer() -> None:
    pg.activate_dp_layout(
        pg.DPLayout(
            rank=3,
            world_size=4,
            group="g",
            src_rank=0,
            writer=True,
            dim_names=("tp",),
            shape=(4,),
        )
    )
    pg.reset_dp_layout()
    assert pg.dp_world_size() == 1
    assert pg.dp_group() is None
