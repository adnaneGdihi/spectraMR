"""The cohort migrator edits text, so what it must not do is the interesting half.

A ``yaml.dump`` round-trip would produce a correct mapping and delete every
comment in these files -- one arm runs to 180 comment lines of rationale. So the
script rewrites lines, and line rewriting is the operation lint cannot protect
you from: it is locally well-formed everywhere it is wrong. The assertions below
are therefore about comment survival, about the arms it must decline, and about
the self-verification catching a mapping that moved.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "migrations"
    / "add_acquired_band_and_coil_barrier.py"
)


@pytest.fixture(scope="module")
def mig():
    spec = importlib.util.spec_from_file_location("_acq_barrier_mig", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARM = """\
model:
  model_type: kspace_cold_diffusion
losses:
  reconstruction:
    # A comment the migration must not eat.
    lambda_pre_dc_kspace: 0.3
    lambda_perceptual: 0.0
  kspace_losses:
  - name: complex_l1
    weight: 1.0
    enabled: true
  - name: sobolev_kspace
    weight: 0.05
    enabled: true
  - name: sense_adjoint_l1
    # Rationale that lives between the name and the weight.
    weight: 0.3
    enabled: true
  complex_losses: []
  policy:
    output_domain: kspace
"""


def _write(tmp_path: Path, text: str = ARM, name: str = "arm.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _run(mig, path: Path, *, apply: bool = True, order: float | None = 6.0) -> str:
    return mig.process(path, apply=apply, barrier=0.25, acquired=0.5, order=order)


class TestItMakesTheIntendedChange:
    def test_all_four_edits_land(self, mig, tmp_path) -> None:
        path = _write(tmp_path)
        verdict = _run(mig, path)
        assert verdict.startswith("APPLIED")

        cfg = yaml.safe_load(path.read_text())["losses"]
        assert cfg["reconstruction"]["lambda_pre_dc_acquired"] == 0.5
        assert cfg["complex_losses"][0]["name"] == "coil_subspace_residual"
        assert cfg["complex_losses"][0]["kwargs"]["input_domain"] == "image"
        by_name = {e["name"]: e for e in cfg["kspace_losses"]}
        assert by_name["sense_adjoint_l1"]["kwargs"] == {"normalize": True}
        assert by_name["sobolev_kspace"]["kwargs"] == {"order": 6.0}

    def test_a_dry_run_writes_nothing(self, mig, tmp_path) -> None:
        path = _write(tmp_path)
        before = path.read_text()
        assert _run(mig, path, apply=False).startswith("WOULD")
        assert path.read_text() == before

    def test_input_domain_is_image_not_kspace(self, mig, tmp_path) -> None:
        """``kspace`` would set ``use_fourier_bridge`` and trip the builder's
        double-bridge guard under ``complex_losses``, which bridges already."""
        path = _write(tmp_path)
        _run(mig, path)
        entry = yaml.safe_load(path.read_text())["losses"]["complex_losses"][0]
        assert entry["kwargs"]["input_domain"] == "image"


class TestItKeepsWhatTheAuthorsWrote:
    def test_every_pre_existing_comment_survives(self, mig, tmp_path) -> None:
        path = _write(tmp_path)
        original = [ln.strip() for ln in ARM.splitlines() if ln.strip().startswith("#")]
        _run(mig, path)
        after = [ln.strip() for ln in path.read_text().splitlines() if ln.strip().startswith("#")]
        assert set(original) <= set(after), "the rewrite ate a comment"

    def test_it_appends_rather_than_reorders(self, mig, tmp_path) -> None:
        path = _write(tmp_path)
        _run(mig, path)
        names = [e["name"] for e in yaml.safe_load(path.read_text())["losses"]["kspace_losses"]]
        assert names == ["complex_l1", "sobolev_kspace", "sense_adjoint_l1"]


class TestItDeclinesWhatItCannotDoCorrectly:
    def test_a_non_cold_diffusion_arm_is_skipped(self, mig, tmp_path) -> None:
        path = _write(tmp_path, ARM.replace("kspace_cold_diffusion", "fdb_baseline"))
        assert _run(mig, path).startswith("SKIP")
        assert "coil_subspace_residual" not in path.read_text()

    def test_an_arm_with_a_different_pre_dc_recipe_is_skipped(self, mig, tmp_path) -> None:
        """Two cohort arms carry no ``lambda_pre_dc_kspace``; there is nothing to
        anchor the sibling to and guessing a home for it would be an invention."""
        path = _write(tmp_path, ARM.replace("    lambda_pre_dc_kspace: 0.3\n", ""))
        assert _run(mig, path).startswith("SKIP")
        assert "lambda_pre_dc_acquired" not in path.read_text()

    def test_an_already_populated_complex_losses_is_skipped(self, mig, tmp_path) -> None:
        path = _write(
            tmp_path,
            ARM.replace(
                "  complex_losses: []\n",
                "  complex_losses:\n  - name: phase_consistency\n    weight: 1.0\n",
            ),
        )
        assert _run(mig, path).startswith("SKIP")

    def test_a_missing_entry_is_reported_not_invented(self, mig, tmp_path) -> None:
        """The two ablation arms that deliberately drop one of these terms must
        keep dropping it -- the ablated axis is the point of the arm."""
        path = _write(
            tmp_path,
            ARM.replace(
                "  - name: sense_adjoint_l1\n"
                "    # Rationale that lives between the name and the weight.\n"
                "    weight: 0.3\n"
                "    enabled: true\n",
                "",
            ),
        )
        verdict = _run(mig, path)
        assert "no-roemer" in verdict
        names = [e["name"] for e in yaml.safe_load(path.read_text())["losses"]["kspace_losses"]]
        assert "sense_adjoint_l1" not in names

    def test_sobolev_is_opt_out(self, mig, tmp_path) -> None:
        """It is the one edit with no measurement behind it, so it has a switch."""
        path = _write(tmp_path)
        _run(mig, path, order=None)
        by_name = {
            e["name"]: e for e in yaml.safe_load(path.read_text())["losses"]["kspace_losses"]
        }
        assert "kwargs" not in by_name["sobolev_kspace"]
        assert by_name["sense_adjoint_l1"]["kwargs"] == {"normalize": True}


class TestTheSelfVerificationIsNotDecorative:
    def test_a_mapping_that_moved_rolls_back(self, mig, tmp_path, monkeypatch) -> None:
        """Planted violation: make one edit write the wrong value and the
        before/after comparison must refuse rather than write the file."""
        path = _write(tmp_path)
        before = path.read_text()

        real = mig._add_acquired_lambda

        def wrong(lines, weight):
            return real(lines, weight * 2)  # a value `_expected` does not predict

        monkeypatch.setattr(mig, "_add_acquired_lambda", wrong)
        assert _run(mig, path).startswith("ROLLBACK")
        assert path.read_text() == before, "rolled back in the verdict but not on disk"
