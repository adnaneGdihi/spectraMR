"""Tests for ``synthesize_pseudo_gt`` — NEX merge + ESPIRiT, with map reuse.

The multi-rep merge is NEX (complex k-space averaging) and the coil maps are
ESPIRiT. For sim->real transfer the single-rep *degraded* must reuse the
NEX-merged reference's maps (one shared coil operator) rather than re-estimate
from a single noisy rep — exercised here via the optional ``smaps`` argument.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import spectramr.infrastructure.physics.coil_sensitivity as csm_mod
import spectramr.infrastructure.physics.m4raw_pseudo_gt as pgt
from spectramr.infrastructure.physics.m4raw_pseudo_gt import extract_contrast_label


class TestExtractContrastLabel:
    """``extract_contrast_label`` must handle both naming schemes (issue #307)."""

    @pytest.mark.parametrize(
        ("stem", "expected"),
        [
            # fastMRI brain — the AX* acquisition token.
            ("file_brain_AXT1_201_2010294", "AXT1"),
            ("file_brain_AXT2_200_2000250", "AXT2"),
            ("file_brain_AXT2FLAIR_201_6002670", "AXT2FLAIR"),
            ("file_brain_AXFLAIR_200_6002462", "AXFLAIR"),
            ("file_brain_AXT1POST_205_2050017", "AXT1POST"),
            ("file_brain_AXT1PRE_205_2050017", "AXT1PRE"),
            # M4Raw — trailing <letters><rep> (must keep working).
            ("2022091411_T101", "T1"),
            ("2022091411_T202", "T2"),
            ("2022091411_FLAIR03", "FLAIR"),
        ],
    )
    def test_known_schemes(self, stem: str, expected: str) -> None:
        assert extract_contrast_label(Path(f"{stem}.h5")) == expected

    def test_fastmri_brain_is_not_returned_as_whole_stem(self) -> None:
        # Pre-fix regression: the AX* stem fell through to ``return stem``.
        label = extract_contrast_label(Path("file_brain_AXT2FLAIR_201_6002670.h5"))
        assert label == "AXT2FLAIR"
        assert "file_brain" not in label

    def test_unknown_scheme_falls_back_to_stem_with_warning(self, caplog) -> None:
        with caplog.at_level("WARNING", logger=pgt.logger.name):
            label = extract_contrast_label(Path("mystery_volume.h5"))
        assert label == "mystery_volume"
        assert any(
            "Could not extract contrast" in r.getMessage() for r in caplog.records
        )


@pytest.fixture()
def fake_kspace(monkeypatch):
    """Patch the H5 loader to return a deterministic (S, C, H, W) k-space."""
    torch.manual_seed(0)
    ks = torch.complex(
        torch.randn(3, 2, 8, 8), torch.randn(3, 2, 8, 8)
    )  # 3 slices, 2 coils

    def _load(_path: Path) -> torch.Tensor:
        return ks

    _stub_h5(monkeypatch, _load)
    return ks


def test_nex_merge_averages_reps_in_kspace(fake_kspace, monkeypatch) -> None:
    # Track the k-space ESPIRiT is handed: it must be the rep-average (NEX), and
    # synthesize must return finite outputs of the right shape.
    seen = {}

    def _fake_espirit(kspace, num_coils, **_):
        seen["kspace"] = kspace
        return torch.ones_like(kspace)

    monkeypatch.setattr(csm_mod, "estimate_csm_espirit", _fake_espirit)
    coil_images, smaps, x_gt_mag, p99 = pgt.synthesize_pseudo_gt(
        [Path("r1.h5"), Path("r2.h5")]
    )
    assert coil_images.shape == (1, 2, 8, 8) and coil_images.is_complex()
    assert smaps.shape == (1, 2, 8, 8)
    assert x_gt_mag.shape == (1, 1, 8, 8)
    assert torch.isfinite(x_gt_mag).all()
    # ESPIRiT saw the NEX-merged k-space (mean over reps of the same slice).
    assert "kspace" in seen and seen["kspace"].shape == (1, 2, 8, 8)


def test_provided_smaps_are_reused_not_reestimated(fake_kspace, monkeypatch) -> None:
    called = {"n": 0}

    def _fake_espirit(kspace, num_coils, **_):
        called["n"] += 1
        return torch.ones_like(kspace)

    monkeypatch.setattr(csm_mod, "estimate_csm_espirit", _fake_espirit)
    ref_smaps = torch.ones(1, 2, 8, 8, dtype=torch.complex64) * 4.0
    _ci, smaps, _mag, _p99 = pgt.synthesize_pseudo_gt([Path("r1.h5")], smaps=ref_smaps)
    # ESPIRiT must NOT run, and the provided maps are returned verbatim.
    assert called["n"] == 0
    assert torch.equal(smaps, ref_smaps)


def test_provided_smaps_shape_mismatch_raises(fake_kspace) -> None:
    bad = torch.ones(1, 5, 8, 8, dtype=torch.complex64)  # wrong coil count
    with pytest.raises(ValueError, match="smaps shape"):
        pgt.synthesize_pseudo_gt([Path("r1.h5")], smaps=bad)


def test_espirit_failure_warns_not_silent(fake_kspace, monkeypatch, caplog) -> None:
    """A non-OOM ESPIRiT failure still degrades to RSS, and says so at ERROR.

    A ground-truth synthesizer that silently swaps ESPIRiT for RSS yields a
    materially different x_gt and corrupts every downstream sim2rank/BT
    comparison with no visible signal (CLAUDE.md pitfall #9/#10). The fallback
    log was promoted DEBUG -> WARNING, and then WARNING -> ERROR once #521
    established that the consequence is a cohort with *two* ground truths.
    """

    def _boom(kspace, num_coils, **_):
        raise RuntimeError("linalg: cusolver unavailable")

    monkeypatch.setattr(csm_mod, "estimate_csm_espirit", _boom)
    with caplog.at_level("ERROR", logger=pgt.logger.name):
        _ci, smaps, x_gt_mag, _p99 = pgt.synthesize_pseudo_gt(
            [Path("r1.h5"), Path("r2.h5")]
        )
    assert smaps.shape == (1, 2, 8, 8)
    assert torch.isfinite(x_gt_mag).all()
    recs = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any(
        "RSS" in r.getMessage() for r in recs
    ), "RSS fallback must be reported at ERROR, not swapped silently"


class TestEspiritOomKeepsOneGroundTruth:
    """#521: a CUDA OOM must change the DEVICE, never the ESTIMATOR.

    On the 2026-07-25 fastMRI-brain run ESPIRiT OOM'd on 7 of 76 volumes and
    each one silently became RSS, so the cohort's leaderboard was computed over
    two incompatible definitions of x_gt with nothing recording which volume
    used which. The OOM itself is not ESPIRiT's fault -- the log shows 23-27 GiB
    already held by the sweep, with ESPIRiT asking for the last ~1.9 GiB -- so
    re-running the same estimator on host RAM keeps the contract intact.
    """

    @staticmethod
    def _oom_once(monkeypatch):
        """ESPIRiT that OOMs on its first call and succeeds thereafter."""
        calls: list[torch.device] = []

        def _espirit(kspace, num_coils, **_):
            calls.append(kspace.device)
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")
            return torch.ones_like(kspace)

        monkeypatch.setattr(csm_mod, "estimate_csm_espirit", _espirit)
        return calls

    def test_oom_retries_espirit_instead_of_substituting_rss(
        self, fake_kspace, monkeypatch
    ) -> None:
        calls = self._oom_once(monkeypatch)
        rss_calls = {"n": 0}

        def _rss(kspace, num_coils, **_):
            rss_calls["n"] += 1
            return torch.ones_like(kspace)

        monkeypatch.setattr(csm_mod, "estimate_csm_rss", _rss)
        _ci, smaps, x_gt_mag, _p99 = pgt.synthesize_pseudo_gt([Path("r1.h5")])

        assert len(calls) == 2, "ESPIRiT must be retried, not abandoned"
        assert rss_calls["n"] == 0, "RSS is a different ground truth — never on OOM"
        assert smaps.shape == (1, 2, 8, 8)
        assert torch.isfinite(x_gt_mag).all()

    def test_retry_runs_on_cpu(self, fake_kspace, monkeypatch) -> None:
        calls = self._oom_once(monkeypatch)
        pgt.synthesize_pseudo_gt([Path("r1.h5")])

        assert calls[1].type == "cpu", "the retry must move the computation to host"

    def test_returned_maps_stay_on_the_input_device(
        self, fake_kspace, monkeypatch
    ) -> None:
        """The retry is an implementation detail; callers see the input device."""
        self._oom_once(monkeypatch)
        coil_images, smaps, _mag, _p99 = pgt.synthesize_pseudo_gt([Path("r1.h5")])

        assert smaps.device == coil_images.device

    def test_oom_is_reported_not_swallowed(
        self, fake_kspace, monkeypatch, caplog
    ) -> None:
        self._oom_once(monkeypatch)
        with caplog.at_level("WARNING", logger=pgt.logger.name):
            pgt.synthesize_pseudo_gt([Path("r1.h5")])

        msgs = [r.getMessage() for r in caplog.records]
        assert any("out of CUDA memory" in m and "CPU" in m for m in msgs)
        assert not any(
            "falling back to RSS" in m for m in msgs
        ), "the OOM path must not claim an RSS substitution"


def test_synthesize_does_not_pin_global_linalg_backend(
    fake_kspace, monkeypatch
) -> None:
    """synthesize_pseudo_gt must NOT mutate the process-global cuSOLVER linalg
    backend.

    The old ``preferred_linalg_library("cusolver")`` pin forced the failing
    batched-complex eigh path (~33 GiB workspace for the (H*W, C, C) Gram →
    CUSOLVER_STATUS_INVALID_VALUE / OOM) and leaked that sticky global setting
    to the whole process. Robustness now lives in
    ``coil_sensitivity._robust_eigh`` (CPU-LAPACK fallback), so no global pin
    is needed — and pinning a process-wide backend from a GT helper is a side
    effect we do not want.
    """
    calls = []

    def _spy(*args):
        calls.append(args)
        return "default"  # mimic the getter form

    monkeypatch.setattr(torch.backends.cuda, "preferred_linalg_library", _spy)

    def _fake_espirit(kspace, num_coils, **_):
        return torch.ones_like(kspace)

    monkeypatch.setattr(csm_mod, "estimate_csm_espirit", _fake_espirit)
    pgt.synthesize_pseudo_gt([Path("r1.h5"), Path("r2.h5")])

    assert calls == [], (
        "synthesize_pseudo_gt must not call preferred_linalg_library; " f"got {calls}"
    )


def test_high_coil_count_grows_acs_to_stay_full_rank(monkeypatch) -> None:
    """A many-coil fully-sampled reference must enlarge the ESPIRiT ACS so the
    calibration stays full-rank instead of silently falling back to RSS (#309).

    fastMRI brain has 12- and 16-coil volumes; the historical 24x24 ACS is
    rank-deficient there (n_patches < kernel^2 * coils), so ESPIRiT would drop to
    RSS and produce a materially different x_gt.
    """
    torch.manual_seed(0)
    n_coils, h, w = 16, 48, 48
    ks = torch.complex(torch.randn(3, n_coils, h, w), torch.randn(3, n_coils, h, w))
    _stub_h5(monkeypatch, lambda _p: ks)

    seen: dict = {}

    def _fake_espirit(kspace, num_coils, **kw):
        seen.update(kw)
        return torch.ones_like(kspace)

    monkeypatch.setattr(csm_mod, "estimate_csm_espirit", _fake_espirit)
    pgt.synthesize_pseudo_gt([Path("r1.h5")])

    acs, kernel = seen["acs_size"], seen["kernel_size"]
    n_patches = (acs - kernel + 1) ** 2
    assert (
        n_patches >= kernel * kernel * n_coils
    ), f"acs_size={acs} still rank-deficient for {n_coils} coils"
    assert acs > 24, "ACS must grow above the 24x24 default for 16 coils"


def _stub_h5(monkeypatch, load_full):
    """Patch BOTH H5 seams from a single full-volume loader.

    ``synthesize_pseudo_gt`` probes each rep's shape from the HDF5 header and
    then reads only the target slice (#616), so a stub that patches only
    ``_load_kspace_from_h5`` leaves the probe hitting the real filesystem —
    which is what these tests did before the two-pass read landed. Deriving
    both seams from one function keeps them consistent by construction rather
    than by two stubs that agree until someone edits one.
    """

    def _shape(path: Path) -> tuple[int, ...]:
        return tuple(load_full(path).shape)

    def _load(path: Path, slice_index: int | None = None):
        volume = load_full(path)
        return volume if slice_index is None else volume[slice_index]

    monkeypatch.setattr(pgt, "_kspace_shape_from_h5", _shape)
    monkeypatch.setattr(pgt, "_load_kspace_from_h5", _load)


def _per_path_loader(coils_by_stem: dict[str, int], h: int = 8, w: int = 8, s: int = 3):
    """Build a ``_load_kspace_from_h5`` stub returning a per-file coil count.

    Genuine NEX repetitions share one receive array, so a group whose reps carry
    *different* coil counts is not a valid NEX set. This lets a test feed exactly
    that malformed group (the [20, C, H, W] vs [16, C, H, W] cluster crash).
    """
    torch.manual_seed(0)

    def _load(path: Path) -> torch.Tensor:
        c = coils_by_stem[path.stem]
        return torch.complex(torch.randn(s, c, h, w), torch.randn(s, c, h, w))

    return _load


def test_heterogeneous_reps_keep_majority_subset_and_warn(monkeypatch, caplog) -> None:
    """A group mixing coil counts must NOT crash ``torch.stack``; it keeps the
    largest coil-consistent subset for the NEX average and WARNS about the drop.

    Regression for the cluster crash ``stack expects each tensor to be equal
    size, but got [20, 768, 396] and [16, 768, 396]``: reps with different coil
    counts are not valid NEX repetitions (coil c of a 20-coil scan is a different
    physical element than coil c of a 16-coil scan), so averaging across the
    mismatch is physically meaningless. The maximal consistent subset is the
    honest NEX set; dropped reps are named at WARNING (no silent facade, #16).
    """
    _stub_h5(monkeypatch, _per_path_loader({"r1": 4, "r2": 4, "r3": 4, "r4": 8}))
    seen: dict = {}

    def _fake_espirit(kspace, num_coils, **_):
        seen["kspace"] = kspace
        return torch.ones_like(kspace)

    monkeypatch.setattr(csm_mod, "estimate_csm_espirit", _fake_espirit)
    with caplog.at_level("WARNING", logger=pgt.logger.name):
        coil_images, smaps, x_gt_mag, _p99 = pgt.synthesize_pseudo_gt(
            [Path("r1.h5"), Path("r2.h5"), Path("r3.h5"), Path("r4.h5")]
        )

    # Majority (3x 4-coil) wins over the lone 8-coil intruder; the average and
    # ESPIRiT both see the 4-coil consistent subset, not the 8-coil rep.
    assert coil_images.shape == (1, 4, 8, 8)
    assert smaps.shape == (1, 4, 8, 8)
    assert x_gt_mag.shape == (1, 1, 8, 8)
    assert torch.isfinite(x_gt_mag).all()
    assert seen["kspace"].shape == (1, 4, 8, 8)
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "r4" in m and ("coil" in m.lower() or "NEX" in m) for m in msgs
    ), f"the dropped 8-coil rep must be named at WARNING; got {msgs}"


def test_two_reps_different_coils_no_crash(monkeypatch, caplog) -> None:
    """The exact 2-rep cluster case (a 1-1 coil-count tie) must not crash.

    With no coil-count majority the reference degrades to a single fully-sampled
    acquisition (a legitimate, lower-SNR reference); the tie is broken toward the
    richer coil encoding and the choice is announced at WARNING, never silent.
    """
    _stub_h5(monkeypatch, _per_path_loader({"r1": 4, "r2": 8}))
    monkeypatch.setattr(
        csm_mod,
        "estimate_csm_espirit",
        lambda kspace, num_coils, **_: torch.ones_like(kspace),
    )
    with caplog.at_level("WARNING", logger=pgt.logger.name):
        coil_images, _smaps, x_gt_mag, _p99 = pgt.synthesize_pseudo_gt(
            [Path("r1.h5"), Path("r2.h5")]
        )
    assert coil_images.shape == (1, 8, 8, 8)  # tie broken toward more coils
    assert torch.isfinite(x_gt_mag).all()
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_low_coil_count_keeps_default_acs(monkeypatch) -> None:
    """Low-coil references (incl. 4-coil M4Raw) keep acs_size=24 unchanged so the
    pseudo-GT stays numerically comparable to prior runs."""
    torch.manual_seed(0)
    n_coils, h, w = 4, 64, 64
    ks = torch.complex(torch.randn(2, n_coils, h, w), torch.randn(2, n_coils, h, w))
    _stub_h5(monkeypatch, lambda _p: ks)

    seen: dict = {}

    def _fake_espirit(kspace, num_coils, **kw):
        seen.update(kw)
        return torch.ones_like(kspace)

    monkeypatch.setattr(csm_mod, "estimate_csm_espirit", _fake_espirit)
    pgt.synthesize_pseudo_gt([Path("r1.h5")])
    assert seen["acs_size"] == 24


class TestReadAcquisitionParams:
    """TR/TE/TI were in the data all along; nothing parsed them (#606).

    ``brc`` and ``qvcr`` reported ``needs acq_params`` and
    ``phys_residual_consistency`` raised *"requires acquisition (TR/TE/TI in ms)"* on
    files whose ``ismrmrd_header`` states ``TR=7500, TE=98, TI=1655, flipAngle=160``
    outright — three metrics declared inapplicable for want of a parser.
    """

    @staticmethod
    def _write(tmp_path, header: str | None):
        import h5py
        import numpy as np

        path = tmp_path / "acq.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("kspace", data=np.zeros((1, 2, 4, 4), dtype=np.complex64))
            if header is not None:
                f.create_dataset("ismrmrd_header", data=np.bytes_(header.encode()))
        return path

    def test_parses_the_namespaced_tags_m4raw_writes(self, tmp_path) -> None:
        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            read_acquisition_params,
        )

        path = self._write(
            tmp_path,
            "<ns0:ismrmrdHeader><ns0:sequenceParameters>"
            "<ns0:TR>7500.0</ns0:TR><ns0:TE>98.0</ns0:TE><ns0:TI>1655.0</ns0:TI>"
            "<ns0:flipAngle_deg>160.0</ns0:flipAngle_deg>"
            "<ns0:sequence_type>FLAIR_TRA</ns0:sequence_type>"
            "</ns0:sequenceParameters></ns0:ismrmrdHeader>",
        )
        got = read_acquisition_params(path)
        assert got["TR"] == 7500.0
        assert got["TE"] == 98.0
        assert got["TI"] == 1655.0
        assert got["flip_angle"] == 160.0
        assert got["sequence_type"] == "FLAIR_TRA"

    def test_parses_an_unnamespaced_header_too(self, tmp_path) -> None:
        """Other vendors write bare ``<TR>``; the regex must not require a prefix."""
        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            read_acquisition_params,
        )

        path = self._write(tmp_path, "<ismrmrdHeader><TR>2500</TR></ismrmrdHeader>")
        assert read_acquisition_params(path)["TR"] == 2500.0

    def test_absent_tag_is_omitted_not_defaulted(self, tmp_path) -> None:
        """A fabricated TR is worse than a missing one — the metric must see the gap."""
        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            read_acquisition_params,
        )

        path = self._write(tmp_path, "<ismrmrdHeader><TE>98.0</TE></ismrmrdHeader>")
        got = read_acquisition_params(path)
        assert got == {"TE": 98.0}
        assert "TR" not in got

    def test_no_header_returns_the_honest_empty(self, tmp_path) -> None:
        """Keeps the metric ``not_applicable`` rather than feeding it invented physics."""
        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            read_acquisition_params,
        )

        assert read_acquisition_params(self._write(tmp_path, None)) == {}

    def test_reads_echo_spacing_train_length_and_encoded_geometry(self, tmp_path) -> None:
        """The series-gain gate and the EPG forward model need these beside TR/TE/TI."""
        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            read_acquisition_params,
        )

        path = self._write(
            tmp_path,
            "<ns0:ismrmrdHeader><ns0:encoding><ns0:encodedSpace><ns0:matrixSize>"
            "<ns0:x>256</ns0:x><ns0:y>256</ns0:y><ns0:z>1</ns0:z></ns0:matrixSize>"
            "<ns0:fieldOfView_mm><ns0:x>240</ns0:x><ns0:y>240</ns0:y><ns0:z>5</ns0:z>"
            "</ns0:fieldOfView_mm></ns0:encodedSpace><ns0:reconSpace><ns0:matrixSize>"
            "<ns0:x>512</ns0:x><ns0:y>512</ns0:y><ns0:z>1</ns0:z></ns0:matrixSize>"
            "</ns0:reconSpace><ns0:echoTrainLength>16</ns0:echoTrainLength></ns0:encoding>"
            "<ns0:sequenceParameters><ns0:TR>4000</ns0:TR><ns0:TE>100</ns0:TE>"
            "<ns0:echo_spacing>12.5</ns0:echo_spacing></ns0:sequenceParameters>"
            "</ns0:ismrmrdHeader>",
        )
        got = read_acquisition_params(path)
        assert got["echo_spacing"] == 12.5
        assert got["echo_train_length"] == 16.0
        assert got["encoded_matrix"] == (256, 256)  # not the reconSpace 512
        assert got["fov_mm"] == (240.0, 240.0, 5.0)

    def test_echo_spacing_with_a_unit_suffix_is_a_number_and_na_is_absent(self, tmp_path) -> None:
        """M4Raw writes ``16.3ms`` on its TSE series and ``N/A`` on its single-echo SE T1."""
        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            read_acquisition_params,
        )

        tse = self._write(tmp_path, "<ismrmrdHeader><echo_spacing>16.3ms</echo_spacing></ismrmrdHeader>")
        assert read_acquisition_params(tse)["echo_spacing"] == 16.3
        se = self._write(tmp_path, "<ismrmrdHeader><echo_spacing>N/A</echo_spacing></ismrmrdHeader>")
        assert "echo_spacing" not in read_acquisition_params(se)

    def test_echo_spacing_in_an_unknown_unit_raises(self, tmp_path) -> None:
        """A silently mis-scaled echo spacing would put the TSE echo at the wrong index."""
        import pytest

        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            read_acquisition_params,
        )

        path = self._write(tmp_path, "<ismrmrdHeader><echo_spacing>0.0163s</echo_spacing></ismrmrdHeader>")
        with pytest.raises(ValueError, match="echo_spacing"):
            read_acquisition_params(path)

    def test_geometry_is_omitted_when_the_header_has_none(self, tmp_path) -> None:
        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            read_acquisition_params,
        )

        got = read_acquisition_params(self._write(tmp_path, "<ismrmrdHeader><TR>1</TR></ismrmrdHeader>"))
        assert "encoded_matrix" not in got and "fov_mm" not in got


class TestSliceOnlyRepReads:
    """#616 (audit D5). Pseudo-GT averages ONE slice across every repetition.

    The target slice is derived from the slice count, so the old single pass
    loaded every rep in FULL to learn a number, held them all resident, then
    used one slice each. On an M4Raw multi-slice multi-coil acquisition that is
    the whole volume per repetition, times the number of repetitions.
    """

    @staticmethod
    def _write_rep(path, n_slices=5, n_coils=4, h=8, w=8, seed=0):
        h5py = pytest.importorskip("h5py")
        import numpy as np

        rng = np.random.default_rng(seed)
        data = rng.standard_normal((n_slices, n_coils, h, w, 2)).astype("float32")
        with h5py.File(path, "w") as f:
            f.create_dataset("kspace", data=data)
        return data

    def test_the_header_probe_reports_the_shape_without_reading(self, tmp_path):
        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            _kspace_shape_from_h5,
        )

        path = tmp_path / "rep0.h5"
        self._write_rep(path)
        assert _kspace_shape_from_h5(path) == (5, 4, 8, 8, 2)

    def test_a_slice_read_equals_the_same_slice_of_a_full_read(self, tmp_path):
        """The equivalence that makes the optimisation safe. Asserted against
        the FULL read rather than against a literal, so a change to the complex
        coercion cannot pass here while breaking one path."""
        import torch

        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            _load_kspace_from_h5,
        )

        path = tmp_path / "rep0.h5"
        self._write_rep(path, n_slices=5)

        full = _load_kspace_from_h5(path)
        for idx in range(5):
            partial = _load_kspace_from_h5(path, slice_index=idx)
            assert torch.equal(partial, full[idx]), f"slice {idx} diverged"

    def test_the_partial_read_drops_the_leading_axis(self, tmp_path):
        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            _load_kspace_from_h5,
        )

        path = tmp_path / "rep0.h5"
        self._write_rep(path, n_slices=5, n_coils=4)
        assert _load_kspace_from_h5(path).shape == (5, 4, 8, 8)
        assert _load_kspace_from_h5(path, slice_index=2).shape == (4, 8, 8)

    def test_a_missing_kspace_key_still_raises_keyerror_on_the_probe(self, tmp_path):
        """The per-rep error handling now spans both passes; the probe must
        raise the same class the loader did so the caller's except arms fire."""
        h5py = pytest.importorskip("h5py")

        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            _kspace_shape_from_h5,
        )

        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("something_else", data=[1, 2, 3])
        with pytest.raises(KeyError):
            _kspace_shape_from_h5(path)

    def test_a_missing_file_raises_filenotfounderror_on_the_probe(self, tmp_path):
        from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
            _kspace_shape_from_h5,
        )

        with pytest.raises(FileNotFoundError):
            _kspace_shape_from_h5(tmp_path / "nope.h5")
