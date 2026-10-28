"""The n2n and self_supervised arms read their coil maps from a complete store.

Under ``coil_maps_missing: estimate`` an empty store does not fail: the loader
runs ESPIRiT on every slice load (0.8-2.3 s single-threaded, #2339). On the
first protocol-v2 runs on sabine that fallback took about 57 % of wall time and
stretched each validation pass to 27-49 min, and nothing reported it beyond one
warning per worker. ``raise`` turns the empty store into a startup error.

The predicate is exercised against a planted violation (non-negotiable 15).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_INPROGRESS = Path(__file__).resolve().parents[3] / "experiments" / "inprogress"
_ARMS = sorted(
    arm
    for cohort in ("n2n", "self_supervised")
    for arm in (_INPROGRESS / cohort).glob("*.yaml")
    if ((yaml.safe_load(arm.read_text()) or {}).get("data") or {}).get("coil_maps_root")
)

#: The arms that read a store on 2026-10-07: 11 n2n, 4 CoNEX, 2 EI multicoil.
_MIN_STORE_ARMS = 17


def _store_policy_error(cfg: dict[str, Any]) -> str | None:
    data = cfg.get("data") or {}
    if not data.get("coil_maps_root"):
        return None
    policy = data.get("coil_maps_missing", "raise")
    if policy != "raise":
        return f"data.coil_maps_missing is {policy!r}; an incomplete store must raise"
    return None


def test_the_corpus_still_has_store_arms() -> None:
    assert len(_ARMS) >= _MIN_STORE_ARMS, [a.stem for a in _ARMS]


@pytest.mark.parametrize("arm", _ARMS, ids=[a.stem for a in _ARMS])
def test_arm_raises_on_a_missing_map_file(arm: Path) -> None:
    error = _store_policy_error(yaml.safe_load(arm.read_text()) or {})
    assert error is None, error


def test_the_predicate_rejects_a_planted_estimate() -> None:
    cfg = yaml.safe_load(_ARMS[0].read_text()) or {}
    assert _store_policy_error(cfg) is None
    cfg["data"]["coil_maps_missing"] = "estimate"
    error = _store_policy_error(cfg)
    assert error is not None and "'estimate'" in error
