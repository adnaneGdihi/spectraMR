"""``tables/_direction.py``: a metric with no direction is shown, never ranked."""

import pytest

from spectramr.infrastructure.reporting.tables._direction import direction


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("psnr", True),  # the SSOT
        ("nmse", False),
        ("cjv_t1", False),  # longest registry token inside the name
        ("my_custom_loss", False),  # a word that only means smaller-is-better
        ("my_custom_score", None),  # nothing says: unranked
    ],
)
def test_direction_sources(metric, expected):
    assert direction(metric) is expected


def test_the_caller_override_wins():
    assert direction("psnr", {"psnr": False}) is False
    assert direction("my_custom_score", {"my_custom_score": True}) is True
