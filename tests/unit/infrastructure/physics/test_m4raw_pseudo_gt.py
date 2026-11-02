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

import spectramr.data.datasets.kspace_volume_stack as kio
import spectramr.data.datasets.pseudo_gt_reference as pgt
import spectramr.infrastructure.physics.coil_sensitivity as csm_mod


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
    with caplog.at_level("ERROR", logger="spectramr"):
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
        with caplog.at_level("WARNING", logger="spectramr"):
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
    ``load_kspace_from_h5`` leaves the probe hitting the real filesystem —
    which is what these tests did before the two-pass read landed. Deriving
    both seams from one function keeps them consistent by construction rather
    than by two stubs that agree until someone edits one.
    """

    def _shape(path: Path) -> tuple[int, ...]:
        return tuple(load_full(path).shape)

    def _load(path: Path, slice_index: int | None = None):
        volume = load_full(path)
        return volume if slice_index is None else volume[slice_index]

    monkeypatch.setattr(kio, "kspace_shape_from_h5", _shape)
    monkeypatch.setattr(kio, "load_kspace_from_h5", _load)


def _per_path_loader(coils_by_stem: dict[str, int], h: int = 8, w: int = 8, s: int = 3):
    """Build a ``load_kspace_from_h5`` stub returning a per-file coil count.

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
    with caplog.at_level("WARNING", logger="spectramr"):
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
    with caplog.at_level("WARNING", logger="spectramr"):
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
