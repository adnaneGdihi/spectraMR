"""``scripts/n2n_excess_bias.py``: per-slice subtraction of the identity's bias floor."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "n2n_excess_bias.py"


@pytest.fixture(scope="module")
def excess() -> Any:
    spec = importlib.util.spec_from_file_location("n2n_excess_bias", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CASES = [f"s{s}/T1/s{s}_T101/{i}" for s in (1, 2) for i in (0, 1, 2)]


def _eval(bias: dict[str, float], var: float = 1.0, nex: float = 4.0) -> dict[str, Any]:
    return {
        "bias_squared": bias,
        "rep_variance": dict.fromkeys(bias, var),
        "reference_noise_corrected_mse": {c: bias[c] + var for c in bias},
        "nex_equivalent": dict.fromkeys(bias, nex),
    }


def _floor() -> dict[str, float]:
    # Planted floor rising with slice index, as on the sabine run.
    return {c: 1.0 + int(c.rsplit("/", 1)[-1]) for c in CASES}


def test_the_floor_is_subtracted_per_slice_not_on_average(excess: Any) -> None:
    identity = _eval(_floor())
    arm = _eval({c: v + 0.5 for c, v in _floor().items()})
    report = excess.build({"arm": arm}, identity)["arms"]["arm"]
    assert report["summary"]["excess_bias_squared"] == pytest.approx(0.5)
    # The slope belongs to the floor, so the excess is flat across slice index.
    assert set(report["by_slice_index"]) == {"0", "1", "2"}
    assert all(v == pytest.approx(0.5) for v in report["by_slice_index"].values())
    assert report["summary"]["nex_equivalent_harmonic"] == pytest.approx(4.0)


def test_an_arm_case_without_an_identity_row_raises(excess: Any) -> None:
    identity = _eval(dict.fromkeys(CASES[:-1], 1.0))
    arm = _eval(dict.fromkeys(CASES, 1.0))
    with pytest.raises(KeyError, match="no identity row"):
        excess.build({"arm": arm}, identity)


def test_a_scalar_final_eval_is_refused(excess: Any, tmp_path: Path) -> None:
    path = tmp_path / "final_eval.json"
    path.write_text(json.dumps({"bias_squared": 0.1}))
    with pytest.raises(ValueError, match="per-case"):
        excess.load(path)


def test_the_cli_writes_a_ranked_table(excess: Any, tmp_path: Path) -> None:
    for name, offset in (("worse", 2.0), ("better", 0.1)):
        (tmp_path / "campaign" / name).mkdir(parents=True)
        payload = _eval({c: v + offset for c, v in _floor().items()})
        (tmp_path / "campaign" / name / "final_eval.json").write_text(json.dumps(payload))
    identity = tmp_path / "identity_final_eval.json"
    identity.write_text(json.dumps(_eval(_floor())))
    out = tmp_path / "out"
    argv = [
        "--campaign",
        str(tmp_path / "campaign"),
        "--identity",
        str(identity),
        "--out",
        str(out),
    ]
    assert excess.main(argv) == 0
    lines = (out / "excess_bias_summary.csv").read_text().splitlines()
    assert [line.split(",")[0] for line in lines[1:]] == ["better", "worse"]
