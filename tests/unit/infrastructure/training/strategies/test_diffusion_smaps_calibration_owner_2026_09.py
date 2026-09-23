"""One owner for the coil-map calibration order, across all three consumers (#2213).

Training, validation and sampling each held a copy of "estimate, RSS-normalise,
match resolution". They had to agree -- a checkpoint that samples with different
maps than it trained with is a silent train/test divergence in the coil geometry
-- and they were kept in step by hand, which is how all three ended up running
the order that collapses RSS between ACS-grid nodes.

These are source scans rather than behavioural tests because the failure they
guard is a FOURTH copy appearing, which no behavioural test of the existing
three can see (non-negotiable 17).
"""

from __future__ import annotations

import inspect
import re

import pytest

from spectramr.infrastructure.inference import cold_diffusion_inference_strategy
from spectramr.infrastructure.physics import coil_sensitivity
from spectramr.infrastructure.training.strategies import diffusion

#: Every module that resolves runtime coil maps for the cold-diffusion family.
CONSUMERS = (diffusion, cold_diffusion_inference_strategy)

#: A local RSS divide over the coil axis -- the first half of the retired order.
_LOCAL_RSS_NORMALISE = re.compile(
    r"sqrt\(\s*\(\s*smaps\.abs\(\)\s*\*\*\s*2\s*\)\.sum\(\s*dim=1", re.MULTILINE
)

#: A local interpolate of a map's real or imaginary half -- the second half.
_LOCAL_MAP_INTERPOLATE = re.compile(r"interpolate\(\s*smaps\.(real|imag)", re.MULTILINE)


@pytest.mark.parametrize("module", CONSUMERS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_consumer_calls_the_owner(module) -> None:
    source = inspect.getsource(module)
    assert "estimate_smaps_calibrated" in source, (
        f"{module.__name__} resolves coil maps without the owner; a second "
        "calibration order is a train/sample divergence nothing reports."
    )


@pytest.mark.parametrize("module", CONSUMERS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_consumer_holds_no_local_normalise_or_resize(module) -> None:
    """The planted shapes: either half of the retired order, re-inlined."""
    source = inspect.getsource(module)
    assert not _LOCAL_RSS_NORMALISE.search(source), (
        f"{module.__name__} RSS-normalises coil maps locally. That divide belongs "
        "to estimate_smaps_calibrated, which runs it AFTER any resize."
    )
    assert not _LOCAL_MAP_INTERPOLATE.search(source), (
        f"{module.__name__} interpolates a coil map's real/imag half locally. "
        "Interpolating the two separately takes the chord between unit-modulus "
        "samples, so the modulus collapses between grid nodes (#2213)."
    )


def test_the_planted_patterns_match_the_retired_code() -> None:
    """A detector is only a detector for a shape it has been watched to catch.

    Verbatim from the three copies this change deleted. Without this, both
    assertions above would keep passing if the regexes silently stopped
    matching anything at all (non-negotiable 15).
    """
    retired = (
        "                rss = torch.sqrt((smaps.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)\n"
        "                smaps = smaps / rss\n"
        "                if smaps.shape[-2:] != (h, w):\n"
        "                    smaps_r = torch.nn.functional.interpolate(\n"
        '                        smaps.real, size=(h, w), mode="bilinear", align_corners=False\n'
        "                    )\n"
    )
    assert _LOCAL_RSS_NORMALISE.search(retired) is not None
    assert _LOCAL_MAP_INTERPOLATE.search(retired) is not None


def test_owner_is_exported_and_shared() -> None:
    """Both consumers bind the same object, not same-named local helpers."""
    assert "estimate_smaps_calibrated" in coil_sensitivity.__all__
    assert (
        diffusion.estimate_smaps_calibrated
        is cold_diffusion_inference_strategy.estimate_smaps_calibrated
        is coil_sensitivity.estimate_smaps_calibrated
    )
