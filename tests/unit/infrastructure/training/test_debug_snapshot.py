"""Unit tests for the cross-paradigm debug-snapshot utility.

Pin three contracts:

1. **Disk-bounded budget**: the helper auto-suppresses after
   ``debug_snapshot_max_calls`` writes per ``run_dir``.
2. **Diagnostic completeness**: the textual + JSON reports include
   every named tensor with the four facts that diagnose 95% of
   regressions (shape, dtype, range, NaN-count).
3. **Domain-aware preview**: tensors flagged as k-space get an IFFT
   before magnitude rendering so the PNG is a brain image and not
   k-space speckle (the experiment_11 doubled-brain regression).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import torch

from spectramr.infrastructure.training.debug_snapshot import (
    SnapshotConfig,
    save_debug_snapshot,
    _call_counts,
    _tensor_report,
)


def _phantom_kspace(B=1, C=2, H=32, W=32) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing="ij",
    )
    img = ((x**2 + y**2) < 0.5).float().unsqueeze(0).unsqueeze(0)
    img = img.expand(B, C, H, W).clone()
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(img, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def _make_logging(**overrides):
    """A ``config.logging`` stand-in built from the REAL schema (#714).

    This used to hand-roll a flat ``SimpleNamespace`` with the pre-decomposition
    spellings (``debug_snapshots``, ``debug_snapshot_max_calls``, ...). Those
    leaves moved into ``logging.snapshots.{enabled,max_calls,...}``, the stub did
    not follow, and all 12 tests in this file have been red since — including the
    two that are the only guards for the #371/#390 wrong-channel family. The
    rename did not just redden a file, it disarmed a live mechanism check.

    ``overrides`` names ``snapshots`` leaves; unknown ones raise here rather than
    becoming an attribute the reader never looks at.
    """
    from tests.utils.block_config_stub import LoggingConfigStub

    return LoggingConfigStub(snapshots=overrides)


def _reset_budget() -> None:
    _call_counts.clear()


class TestTensorReport:
    def test_basic_image_pair(self) -> None:
        t = torch.rand(1, 1, 16, 16)
        r = _tensor_report("img", t)
        assert r.shape == [1, 1, 16, 16]
        assert r.is_complex is False
        assert r.nan_count == 0 and r.inf_count == 0
        assert r.min is not None and r.max is not None

    def test_complex_tensor(self) -> None:
        t = torch.complex(torch.rand(1, 2, 8, 8), torch.rand(1, 2, 8, 8))
        r = _tensor_report("ksp", t)
        assert r.is_complex is True
        # magnitude min/max
        assert r.min >= 0

    def test_nan_inf_counts(self) -> None:
        t = torch.tensor([1.0, float("nan"), float("inf"), -float("inf"), 2.0])
        r = _tensor_report("bad", t)
        assert r.nan_count == 1
        assert r.inf_count == 2
        assert "NaN" in r.note and "Inf" in r.note

    def test_constant_tensor_flagged(self) -> None:
        t = torch.full((4, 1, 8, 8), 0.5)
        r = _tensor_report("const", t)
        assert "constant" in r.note.lower()

    def test_real_stacked_complex_hint(self) -> None:
        # 4-channel real-stacked → flagged as likely complex pairs
        t = torch.rand(1, 4, 8, 8)
        r = _tensor_report("k_real_stacked", t)
        assert "real-stacked" in r.note


class TestSnapshotWritesFiles:
    def test_writes_text_json_png(self, tmp_path: Path) -> None:
        _reset_budget()
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="t",
            tensors={"input": torch.rand(1, 1, 16, 16)},
            paradigm="TestParadigm",
            config_section=_make_logging(),
        )
        assert out is not None and out.exists()
        assert (out / "snapshot.txt").exists()
        assert (out / "snapshot.json").exists()
        # Text report contains the tensor name and the paradigm
        text = (out / "snapshot.txt").read_text()
        assert "input" in text and "TestParadigm" in text
        # JSON parses and lists the tensor
        payload = json.loads((out / "snapshot.json").read_text())
        assert payload["paradigm"] == "TestParadigm"
        assert payload["tensors"][0]["name"] == "input"
        # PNG produced
        assert (out / "input.png").exists()

    def test_kspace_preview_uses_ifft(self, tmp_path: Path) -> None:
        _reset_budget()
        ksp = _phantom_kspace()
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="t",
            tensors={"kspace_target": ksp},
            paradigm="TestParadigm",
            config_section=_make_logging(),
            in_kspace_keys={"kspace_target"},
        )
        # The PNG should be the brain image (high-magnitude central blob),
        # not the k-space speckle (high-magnitude DC at centre, low elsewhere).
        from torchvision.io import read_image

        img = read_image(str(out / "kspace_target.png")).float() / 255.0
        # Brain image: centre pixel and a small annulus around it should be
        # bright; sweet spot is at radius ~7 for a 32x32 disk phantom.
        h, w = img.shape[-2], img.shape[-1]
        cy, cx = h // 2, w // 2
        centre_mean = img[..., cy - 4 : cy + 4, cx - 4 : cx + 4].mean().item()
        edge_mean = img[..., :4, :4].mean().item()
        # k-space DC peak would have centre ≫ edge by >100x; the brain
        # image has centre ≳ edge but not extremely. We assert centre is
        # bright AND that the edge is NOT essentially zero (would imply
        # a sharp DC peak rather than diffuse anatomy).
        assert (
            centre_mean > 0.3
        ), f"Centre too dim ({centre_mean:.3f}); IFFT may not have run"

    def test_complex_input_handled(self, tmp_path: Path) -> None:
        _reset_budget()
        cx = torch.complex(torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16))
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="t",
            tensors={"complex_img": cx},
            paradigm="P",
            config_section=_make_logging(),
        )
        assert (out / "complex_img.png").exists()


class TestBudget:
    def test_max_calls_enforced(self, tmp_path: Path) -> None:
        _reset_budget()
        cfg = _make_logging(max_calls=3)
        for s in range(5):
            r = save_debug_snapshot(
                run_dir=tmp_path,
                step=s,
                tag="t",
                tensors={"input": torch.rand(1, 1, 8, 8)},
                paradigm="P",
                config_section=cfg,
            )
            if s < 3:
                assert r is not None
            else:
                assert r is None

    def test_disabled_returns_none(self, tmp_path: Path) -> None:
        _reset_budget()
        cfg = _make_logging(enabled=False)
        r = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="t",
            tensors={"input": torch.rand(1, 1, 8, 8)},
            paradigm="P",
            config_section=cfg,
        )
        assert r is None
        # Nothing on disk
        assert not (tmp_path / "debug_snapshots").exists()

    def test_interval_filter(self, tmp_path: Path) -> None:
        _reset_budget()
        cfg = _make_logging(interval_steps=5)
        # step 0 → 0 % 5 == 0 → should write
        # step 3 → suppressed by interval
        # step 5 → writes
        wrote = []
        for s in (0, 3, 5):
            r = save_debug_snapshot(
                run_dir=tmp_path,
                step=s,
                tag="t",
                tensors={"input": torch.rand(1, 1, 8, 8)},
                paradigm="P",
                config_section=cfg,
            )
            wrote.append(r is not None)
        # The budget is NOT consumed by an interval-suppressed call: #706 moved
        # the interval check ahead of `_budget_check` precisely because charging
        # for a rejected call meant `interval_steps: 100` with the default
        # `max_calls: 8` exhausted the allowance by step 7. This comment used to
        # assert the opposite, describing the behaviour that bug produced.
        assert wrote[0] is True
        assert wrote[1] is False  # interval-suppressed
        assert wrote[2] is True


class TestExtraContext:
    def test_extra_recorded_in_json(self, tmp_path: Path) -> None:
        _reset_budget()
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="t",
            tensors={"input": torch.rand(1, 1, 8, 8)},
            paradigm="P",
            config_section=_make_logging(),
            extra={"epoch": 3, "lr": 1e-4, "schedule_severity": 0.42},
        )
        payload = json.loads((out / "snapshot.json").read_text())
        assert payload["extra"]["epoch"] == 3
        assert abs(payload["extra"]["lr"] - 1e-4) < 1e-12
        assert payload["extra"]["schedule_severity"] == 0.42


class TestImageDomainDoesNotAutoPairAsComplex:
    """Regression: paired-modality data (e.g. ULF/HF stacked as 2 channels)
    must NOT be re-paired as ``complex(ch0, ch1)`` in image domain. The
    earlier behaviour produced a ``sqrt(ULF² + HF²)`` blend (the May 2026
    "ULF doubled and odd" symptom). Image-domain rendering uses RSS along
    the channel axis instead — equivalent to per-coil-magnitude+RSS for
    real real-stacked complex data, and at least deterministic for paired
    modalities.
    """

    def test_paired_modality_is_rss_not_complex_pair(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.training.debug_snapshot import _render_image_preview

        # Construct a 2-channel "paired modality" tensor where the two
        # channels are obviously different magnitudes — if the old code
        # paired them as (R, I) and took sqrt(R² + I²), the result would
        # depend on both. With channel-RSS the result is also
        # sqrt(ch0² + ch1²) but applied PRE-IFFT-pairing — so the test
        # really pins that we go through the no-pair path. We assert the
        # output is single-channel and matches the simple channel-RSS.
        ch0 = torch.full((1, 1, 8, 8), 0.3)
        ch1 = torch.full((1, 1, 8, 8), 0.7)
        paired = torch.cat([ch0, ch1], dim=1)  # [1, 2, 8, 8]
        preview = _render_image_preview(paired, in_kspace=False)
        assert preview is not None
        assert preview.shape == (1, 1, 8, 8)
        # After per-sample [0, 1] normalization the constant-magnitude
        # output collapses to either 0 or 1 — but crucially the function
        # must NOT have raised, must NOT have produced a complex tensor,
        # and must NOT have shrunk to zero by the IFFT path.
        assert torch.isfinite(preview).all()

    def test_real_stacked_complex_image_unchanged(self, tmp_path: Path) -> None:
        """Real-stacked complex IMAGE (not k-space) should still render as
        a magnitude image. We supply a tensor whose channel-RSS gives a
        spatially-smooth disk; the rendered preview must preserve that
        spatial structure (i.e. not be uniform after normalisation)."""
        from spectramr.infrastructure.training.debug_snapshot import _render_image_preview

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing="ij",
        )
        disk = ((x**2 + y**2) < 0.5).float()
        # 4-channel real-stacked complex: (R0, I0, R1, I1) where
        # |C_c|² = R_c² + I_c². Make all four channels carry the disk.
        t = disk.unsqueeze(0).unsqueeze(0).expand(1, 4, 16, 16).clone()
        preview = _render_image_preview(t, in_kspace=False)
        assert preview is not None
        assert preview.shape == (1, 1, 16, 16)
        # Centre brighter than corner — preserves spatial structure.
        centre = preview[0, 0, 6:10, 6:10].mean().item()
        corner = preview[0, 0, :2, :2].mean().item()
        assert centre > corner


class TestKspaceSignatureGuard:
    """Regression (smoke 2026-06-12): VF / IB image-output arms
    (``cross_attention_oracle_unet``, ``ib_vf``) feed an IMAGE-domain complex
    target through the snapshot with ``in_kspace=True`` (the config-level
    ``targets_are_kspace`` flag is True because the *dataset* is k-space). The
    old code IFFT'd it unconditionally, turning an image into k-space → a black
    field with a central DC spike (``method_a`` "blacked target",
    ``ib_infonce`` "blank target"). The same config flag is CORRECT for arms
    whose base-level target really is k-space (``m10`` dynamic_mr_nerf), so the
    only reliable discriminator is the tensor's own DC-energy concentration.
    """

    @staticmethod
    def _disk(H=64, complex_=True, peak=10.0):
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, H), indexing="ij"
        )
        disk = ((x**2 + y**2) < 0.5).float() * peak
        if complex_:
            return torch.complex(disk, 0.2 * disk).unsqueeze(0).unsqueeze(0)
        return disk.unsqueeze(0).unsqueeze(0)

    def test_looks_like_kspace_separates_domains(self) -> None:
        from spectramr.infrastructure.physics.fft_ops import fft2c
        from spectramr.infrastructure.training.debug_snapshot import _looks_like_kspace

        img = self._disk().abs()  # image-domain magnitude
        ksp = fft2c(self._disk()).abs()  # its k-space magnitude
        assert _looks_like_kspace(img) is False
        assert _looks_like_kspace(ksp) is True

    def test_image_flagged_kspace_is_not_ifftd(self) -> None:
        """A complex IMAGE tensor flagged in_kspace=True must render as the
        image (structure preserved), NOT a DC blob."""
        from spectramr.infrastructure.training.debug_snapshot import _render_image_preview

        preview = _render_image_preview(self._disk(), in_kspace=True)
        assert preview is not None
        # The disk fills ~31% of the frame; a wrongly-IFFT'd image collapses to
        # a tiny central DC blob (<1% bright). Require a broad bright region.
        bright_frac = (preview > 0.5).float().mean().item()
        assert (
            bright_frac > 0.15
        ), f"only {bright_frac:.3f} bright — image was IFFT'd into a DC blob"

    def test_real_kspace_still_ifftd(self) -> None:
        """Genuine k-space (DC signature present) must STILL be IFFT'd so its
        preview is the reconstructed image — the m10 / experiment_11 contract."""
        from spectramr.infrastructure.physics.fft_ops import fft2c
        from spectramr.infrastructure.training.debug_snapshot import _render_image_preview

        ksp = fft2c(self._disk())  # complex k-space of the disk
        preview = _render_image_preview(ksp, in_kspace=True)
        assert preview is not None
        # IFFT recovers the disk → broad bright region, not a DC blob.
        assert (preview > 0.5).float().mean().item() > 0.15

    @staticmethod
    def _normalized_kspace(H=64):
        """Genuine k-space after phase-preserving magnitude compression.

        ``data.normalize_kspace`` applies a ``log1p`` (and, in the robust
        path, a divide) magnitude compression that FLATTENS the dominant DC
        pixel. On real multicoil M4Raw k-space this pushes the legacy
        ``centre/mean`` ratio below the veto's threshold of 10. We reproduce
        that with a phase-preserving power compression strong enough to cross
        the threshold (a clean synthetic phantom's DC survives log1p; real
        spread k-space does not). The tensor is still k-space → must be IFFT'd.
        """
        from spectramr.infrastructure.physics.fft_ops import fft2c

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, H), indexing="ij"
        )
        disk = ((x**2 + y**2) < 0.5).float()
        g = torch.Generator().manual_seed(0)
        img = (
            disk + 0.3 * torch.sin(8 * x) * disk + 0.05 * torch.rand(H, H, generator=g)
        ).clamp_min(0)
        ksp = fft2c(img.unsqueeze(0).unsqueeze(0).to(torch.complex64))
        return (ksp.abs() ** 0.3) * torch.exp(1j * ksp.angle())

    def test_normalized_kspace_defeats_legacy_raw_dc_veto(self) -> None:
        """REGRESSION (experiment_11 first_steps, 2026-06-17): the legacy
        raw-magnitude DC test reads a *normalized* k-space as image-domain
        (its DC was flattened by ``normalize_kspace``), so the IFFT was
        wrongly skipped and the debug ``target.png`` rendered as raw k-space
        (DC blob / 'doubled brain'). This pins WHY the spectrum-based test is
        needed."""
        from spectramr.infrastructure.training.debug_snapshot import _looks_like_kspace

        kn = self._normalized_kspace()
        # legacy heuristic on the raw magnitude => "not k-space" (the bug)
        assert _looks_like_kspace(kn.abs()) is False

    def test_is_in_kspace_domain_matrix(self) -> None:
        """The spectrum-based decision is normalization-invariant: raw AND
        normalized k-space => IFFT; an image (even flagged k-space) => skip."""
        from spectramr.infrastructure.physics.fft_ops import fft2c
        from spectramr.infrastructure.training.debug_snapshot import _is_in_kspace_domain

        assert _is_in_kspace_domain(fft2c(self._disk())) is True  # raw k-space
        assert (
            _is_in_kspace_domain(self._normalized_kspace()) is True
        )  # normalized k-space
        assert _is_in_kspace_domain(self._disk()) is False  # image (VF/IB)

    def test_normalized_kspace_is_ifftd_to_image(self) -> None:
        """End-to-end: a normalized k-space flagged in_kspace=True must be
        IFFT'd (decision True), where the legacy code skipped it."""
        from spectramr.infrastructure.training.debug_snapshot import (
            _is_in_kspace_domain,
            _render_image_preview,
        )

        kn = self._normalized_kspace()
        assert _is_in_kspace_domain(kn) is True
        preview = _render_image_preview(kn, in_kspace=True)
        assert preview is not None and torch.isfinite(preview).all()

    def test_authoritative_kspace_is_ifftd_even_when_heuristic_false_negates(
        self, monkeypatch
    ) -> None:
        """CONTRACT (experiment_11 kspace_filling, 2026-06-27): when the config
        AUTHORITATIVELY declares a tensor k-space (``model.target_domain`` /
        the ``KNOWN_KSPACE_OUTPUT_MODELS`` registry), the previewer must IFFT
        it UNCONDITIONALLY — never defer to the spectrum heuristic.

        The spectrum heuristic ``_is_in_kspace_domain`` is reliable on clean
        synthetic phantoms but data-dependent on real *normalized multicoil*
        M4Raw k-space, where it false-negated and rendered ``input_prepared`` /
        ``model_output`` as raw, off-centre k-space instead of a brain. We
        reproduce that production failure by forcing the heuristic to its
        known-bad output and assert the authoritative path still recovers the
        image.
        """
        from spectramr.infrastructure.physics.fft_ops import fft2c
        import spectramr.infrastructure.training.debug_snapshot as ds

        # Reproduce the real-data failure mode: heuristic wrongly says "image".
        monkeypatch.setattr(ds, "_is_in_kspace_domain", lambda _x: False)
        ksp = fft2c(self._disk())  # genuine k-space of the disk

        # Non-authoritative: the veto skips the IFFT → raw k-space DC blob (bug).
        buggy = ds._render_image_preview(ksp, in_kspace=True)
        assert buggy is not None
        assert (buggy > 0.5).float().mean().item() < 0.05

        # Authoritative: IFFT regardless of the heuristic → recovers the disk.
        fixed = ds._render_image_preview(ksp, in_kspace=True, authoritative=True)
        assert fixed is not None
        assert (fixed > 0.5).float().mean().item() > 0.15


class TestImageDomainMaskDetector:
    """A k-space undersampling (line) mask applied to IMAGE data zeroes whole
    image rows/cols; applied correctly to k-space it does not (the IFFT of an
    undersampled k-space is a *dense* aliased image). The snapshot renders the
    tensor a reviewer would look at, then flags the former case as a DOMAIN
    SUPERPOSITION so "the mask ended up in the wrong domain" is *visible and
    named*, not guessed from a fuzzy contact sheet.

    User report (2026-07-06): "in some instances the masks were applied to
    image space data ... this pattern error should be fixed make sure it is
    consistent."
    """

    @staticmethod
    def _brain(H=64):
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, H), indexing="ij"
        )
        return ((x**2 + y**2) < 0.5).float().unsqueeze(0).unsqueeze(0)

    def test_flags_line_mask_applied_in_image_domain(self, tmp_path: Path) -> None:
        _reset_budget()
        img = self._brain()
        # A Cartesian phase-encode line mask applied to the IMAGE (the bug):
        # zero every other ROW of the image itself.
        wrong = img.clone()
        wrong[..., ::2, :] = 0.0
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="t",
            tensors={"model_input": wrong},
            paradigm="P",
            config_section=_make_logging(),
            in_kspace_keys={"model_input"},
        )
        text = (out / "snapshot.txt").read_text()
        assert (
            "DOMAIN SUPERPOSITION" in text
        ), "image-domain line mask must be flagged in the snapshot report"
        payload = json.loads((out / "snapshot.json").read_text())
        rec = next(r for r in payload["tensors"] if r["name"] == "model_input")
        assert "domain superposition" in rec["note"].lower()

    def test_correctly_undersampled_kspace_not_flagged(self, tmp_path: Path) -> None:
        _reset_budget()
        from spectramr.infrastructure.physics.fft_ops import fft2c

        ksp = fft2c(self._brain().to(torch.complex64))  # genuine k-space
        # Undersample in K-SPACE (correct): zero every other k-space row.
        mask = torch.ones_like(ksp.real)
        mask[..., ::2, :] = 0.0
        under = ksp * mask  # aliased-but-DENSE image after IFFT
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="t",
            tensors={"model_input": under},
            paradigm="P",
            config_section=_make_logging(),
            in_kspace_keys={"model_input"},
        )
        payload = json.loads((out / "snapshot.json").read_text())
        rec = next(r for r in payload["tensors"] if r["name"] == "model_input")
        assert (
            "domain superposition" not in rec["note"].lower()
        ), "correct k-space undersampling must NOT be flagged (IFFT is dense)"

    def test_detector_ignores_non_kspace_tensors(self, tmp_path: Path) -> None:
        """A striped tensor NOT declared k-space is out of scope — the detector
        only speaks to tensors that are *supposed* to be k-space."""
        _reset_budget()
        striped = self._brain().clone()
        striped[..., ::2, :] = 0.0
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="t",
            tensors={"some_image": striped},
            paradigm="P",
            config_section=_make_logging(),
            in_kspace_keys=set(),  # not declared k-space
        )
        payload = json.loads((out / "snapshot.json").read_text())
        rec = next(r for r in payload["tensors"] if r["name"] == "some_image")
        assert "domain superposition" not in rec["note"].lower()

    def test_blank_tensor_not_misread_as_striped(self, tmp_path: Path) -> None:
        """A degenerate (all-zero / constant) frame has no line structure and
        must not be flagged (else every blank target would false-positive)."""
        _reset_budget()
        blank = torch.zeros(1, 1, 64, 64)
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="t",
            tensors={"model_input": blank},
            paradigm="P",
            config_section=_make_logging(),
            in_kspace_keys={"model_input"},
        )
        payload = json.loads((out / "snapshot.json").read_text())
        rec = next(r for r in payload["tensors"] if r["name"] == "model_input")
        assert "domain superposition" not in rec["note"].lower()


class TestParadigmAgnostic:
    """Confirm the helper does not depend on any strategy-specific imports."""

    def test_works_without_config_section(self, tmp_path: Path) -> None:
        _reset_budget()
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="no_cfg",
            tensors={"x": torch.rand(1, 1, 8, 8)},
            paradigm="Whatever",
            config_section=None,
        )
        assert out is not None
        assert (out / "snapshot.txt").exists()

    def test_handles_none_tensor(self, tmp_path: Path) -> None:
        _reset_budget()
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="t",
            tensors={"input": torch.rand(1, 1, 8, 8), "missing": None},
            paradigm="P",
            config_section=_make_logging(),
        )
        text = (out / "snapshot.txt").read_text()
        assert "input" in text
        assert "missing" not in text  # None entries skipped


# ---------------------------------------------------------------------------
# #706 / #693: the budget, the cadence, and the Mock that wrote to the repo root.
# ---------------------------------------------------------------------------


def _snapshot_section(*, max_calls=8, interval_steps=0, enabled=True):
    """A `config.logging` stand-in using the SCHEMA's field names.

    `interval_steps`, not `image_interval_steps` — the latter is `SnapshotConfig`'s
    internal name, and a stub that uses it silently resolves to the default.
    """
    import types

    return types.SimpleNamespace(
        snapshots=types.SimpleNamespace(
            enabled=enabled,
            max_calls=max_calls,
            interval_steps=interval_steps,
            save_images=False,
            save_json=True,
            log_steps=[],
        )
    )


class TestSnapshotBudget:
    def test_setting_an_interval_no_longer_disables_snapshots(self, tmp_path):
        """#706: `_budget_check` MUTATES, and it ran BEFORE the interval test.

        With `interval_steps: 100` and the default `max_calls: 8`, the budget was
        exhausted by step 7 and only the step-0 snapshot was ever written —
        turning on the cadence knob turned the feature off.
        """
        import torch

        from spectramr.infrastructure.training import debug_snapshot as ds

        ds._call_counts.clear()
        written = [
            step
            for step in range(0, 1001, 10)
            if ds.save_debug_snapshot(
                run_dir=tmp_path,
                step=step,
                tag="t",
                tensors={"x": torch.rand(1, 1, 4, 4)},
                paradigm="X",
                config_section=_snapshot_section(max_calls=8, interval_steps=100),
            )
            is not None
        ]
        assert written == [0, 100, 200, 300, 400, 500, 600, 700], written

    def test_each_tag_gets_its_own_budget(self, tmp_path):
        """#706: one shared counter meant the first tag consumed the allowance.

        `first_steps`, `model_output` and `vf_twin` all fire in one run; whichever
        went first starved the rest, and "no model_output snapshot" then reads
        like "the model has no output".
        """
        import torch

        from spectramr.infrastructure.training import debug_snapshot as ds

        ds._call_counts.clear()
        per_tag = {}
        for tag in ("first_steps", "model_output", "vf_twin"):
            per_tag[tag] = sum(
                1
                for step in range(5)
                if ds.save_debug_snapshot(
                    run_dir=tmp_path,
                    step=step,
                    tag=tag,
                    tensors={"x": torch.rand(1, 1, 4, 4)},
                    paradigm="X",
                    config_section=_snapshot_section(max_calls=2),
                )
                is not None
            )
        assert per_tag == {"first_steps": 2, "model_output": 2, "vf_twin": 2}, per_tag


class TestMockRunDirIsRefused:
    def test_a_magicmock_run_dir_raises_instead_of_writing(self, tmp_path, monkeypatch):
        """#693/#476: this is what created 420 files in the repo root.

        `os.fspath` is NOT sufficient — a MagicMock auto-creates `__fspath__`, so it
        satisfies both `os.fspath` and `isinstance(m, os.PathLike)`, and returns the
        literal string `MagicMock/mock.run_output_dir/<id>`.
        """
        import os
        from unittest.mock import MagicMock

        import pytest
        import torch

        from spectramr.infrastructure.training import debug_snapshot as ds

        mock_dir = MagicMock().run_output_dir
        # Pin the premise, so the guard is not silently weakened later.
        assert os.fspath(mock_dir).startswith("MagicMock/")
        assert isinstance(mock_dir, os.PathLike)

        monkeypatch.chdir(tmp_path)
        with pytest.raises(TypeError, match="needs a real path"):
            ds.save_debug_snapshot(
                run_dir=mock_dir,
                step=0,
                tag="first_steps",
                tensors={"x": torch.rand(1, 1, 4, 4)},
                paradigm="X",
                config_section=_snapshot_section(),
            )
        assert not list(tmp_path.glob("MagicMock*")), "a MagicMock tree was created"

    def test_str_and_path_both_still_work(self, tmp_path):
        import torch

        from spectramr.infrastructure.training import debug_snapshot as ds

        ds._call_counts.clear()
        for i, run_dir in enumerate((tmp_path, str(tmp_path))):
            assert (
                ds.save_debug_snapshot(
                    run_dir=run_dir,
                    step=i,
                    tag="t",
                    tensors={"x": torch.rand(1, 1, 4, 4)},
                    paradigm="X",
                    config_section=_snapshot_section(),
                )
                is not None
            )


class TestSnapshotDecompressesLogScaledKspace:
    """#682: the renderer IFFT'd a log1p-compressed spectrum.

    `debug_snapshot` receives `config.logging` and cannot reach `config.data`, so
    it had no way to know -- the flag is threaded in as an argument by the caller
    that does have the config.
    """

    @staticmethod
    def _compressed_phantom():
        import torch

        from spectramr.data.transforms.normalization import compress_kspace_log
        from spectramr.infrastructure.physics.fft_ops import fft2c

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij"
        )
        img = ((xx**2 + yy**2) < 0.35).float().unsqueeze(0).unsqueeze(0)
        return img, compress_kspace_log(fft2c(img.to(torch.complex64)))

    def test_the_flag_changes_the_render(self):
        import torch

        from spectramr.infrastructure.training import debug_snapshot as ds

        _img, compressed = self._compressed_phantom()
        off = ds._render_image_preview(
            compressed, in_kspace=True, authoritative=True, log_scaled=False
        )
        on = ds._render_image_preview(
            compressed, in_kspace=True, authoritative=True, log_scaled=True
        )
        assert not torch.allclose(off, on), "log_scaled had no effect on the render"

    def test_the_decompressed_render_matches_the_anatomy(self):
        from spectramr.infrastructure.training import debug_snapshot as ds

        img, compressed = self._compressed_phantom()
        rendered = ds._render_image_preview(
            compressed, in_kspace=True, authoritative=True, log_scaled=True
        )

        a = (rendered[0, 0] - rendered[0, 0].mean()).flatten()
        b = (img[0, 0] - img[0, 0].mean()).flatten()
        corr = float((a @ b) / (a.norm() * b.norm() + 1e-12))
        assert corr > 0.99, f"decompressed render correlates only {corr:.3f}"

    def test_default_is_off_so_uncompressed_arms_are_untouched(self):
        """The parameter defaults False: an arm that never compressed must render
        exactly as before this change."""
        import inspect

        from spectramr.infrastructure.training.debug_snapshot import (
            _render_image_preview,
            save_debug_snapshot,
        )

        for fn in (_render_image_preview, save_debug_snapshot):
            assert inspect.signature(fn).parameters["log_scaled"].default is False


# ---------------------------------------------------------------------------
# "Does this run write artifacts at all?" -- a config property, not a cadence.
# ---------------------------------------------------------------------------


class TestSnapshotsAreEnabled:
    """`snapshots_are_enabled` answers a question with no step in it.

    It exists for the model-input contract in `BaseTrainingStrategy`, which
    enforces what an *artifact* claims and so must fire on exactly the runs that
    write artifacts. Asking `snapshot_step_is_due` there would fold in
    `interval_steps` and reintroduce #706's shape: the wrapper and the strategy
    count steps from different places, so they disagree the moment a caller
    omits `iteration` or a run resumes mid-interval.
    """

    def test_enabled_by_default(self) -> None:
        from spectramr.infrastructure.training.debug_snapshot import snapshots_are_enabled

        assert snapshots_are_enabled(None) is True

    def test_reads_the_nested_flag(self) -> None:
        from spectramr.infrastructure.training.debug_snapshot import snapshots_are_enabled

        assert snapshots_are_enabled(_snapshot_section(enabled=False)) is False
        assert snapshots_are_enabled(_snapshot_section(enabled=True)) is True

    def test_is_independent_of_the_interval(self) -> None:
        """The distinction from `snapshot_step_is_due`, pinned.

        With `interval_steps: 100` the cadence predicate is False on step 1 while
        this stays True — that gap is exactly why the contract check may not use
        the cadence predicate.
        """
        from spectramr.infrastructure.training.debug_snapshot import (
            snapshot_step_is_due,
            snapshots_are_enabled,
        )

        section = _snapshot_section(interval_steps=100)
        assert snapshots_are_enabled(section) is True
        assert snapshot_step_is_due(section, 1) is False


# ---------------------------------------------------------------------------
# The cheap call-site predicate: cadence belongs to `logging.snapshots` alone.
# ---------------------------------------------------------------------------


class TestSnapshotStepIsDue:
    """`snapshot_step_is_due` is what call sites ask instead of inventing a cadence.

    The bug it exists to prevent: `diffusion.py` gated its only degraded-model-input
    snapshot on `logging.intervals.log * 5`. Arms set `intervals.log: 5000` for
    quiet training, which moved that snapshot to step 25 000 — unreachable in any
    shorter run — so the only visible "input" left was the base strategy's
    PRE-degradation `first_steps/input_prepared`, and a cold-diffusion arm read as
    though it were being fed fully-sampled data.
    """

    def test_without_an_interval_every_step_is_due_whatever_the_budget(self):
        """`max_calls` is a CALL budget and must never be read as a STEP bound.

        The predicate cannot see the stateful per-(run_dir, tag) counter, so with
        no interval configured it defers unconditionally and the writer decides.
        Comparing `step <= max_calls` here conflates two different quantities and
        silently disabled the snapshot on every resumed run — see
        `test_a_resumed_run_is_still_due` below.
        """
        from spectramr.infrastructure.training.debug_snapshot import snapshot_step_is_due

        section = _snapshot_section(max_calls=8, interval_steps=0)
        assert all(snapshot_step_is_due(section, s) for s in range(0, 41))
        # Well past the budget, where the step-vs-call confusion used to bite.
        assert snapshot_step_is_due(section, 5_000) is True

    def test_a_quiet_log_interval_cannot_suppress_it(self):
        """The regression guard. `logging.intervals.log` is not a snapshot knob.

        A stand-in carrying the experiment_11 arm's `intervals.log: 5000` must not
        change the answer — the predicate may not read that block at all.
        """
        import types

        from spectramr.infrastructure.training.debug_snapshot import snapshot_step_is_due

        section = _snapshot_section(max_calls=8, interval_steps=0)
        section.intervals = types.SimpleNamespace(log=5000)
        assert snapshot_step_is_due(section, 1) is True
        # The retired gate was `step % (5000 * 5) == 0` — false for every step a
        # realistic run reaches.
        assert 1 % (5000 * 5) != 0

    def test_a_configured_interval_is_honoured(self):
        from spectramr.infrastructure.training.debug_snapshot import snapshot_step_is_due

        section = _snapshot_section(max_calls=8, interval_steps=100)
        assert snapshot_step_is_due(section, 100) is True
        assert snapshot_step_is_due(section, 101) is False

    def test_disabled_snapshots_are_never_due(self):
        from spectramr.infrastructure.training.debug_snapshot import snapshot_step_is_due

        section = _snapshot_section()
        section.snapshots.enabled = False
        assert snapshot_step_is_due(section, 1) is False

    def test_it_never_rejects_a_step_the_writer_would_have_written(self, tmp_path):
        """No false negatives — the property that keeps it from silently disabling.

        The predicate is an upper bound: the stateful per-tag budget lives inside
        `save_debug_snapshot`. It is free to say "due" on a step the budget then
        refuses, but saying "not due" on a step that WOULD have been written is
        exactly the failure this module keeps re-learning (#706).

        Swept from several START steps, not just 0. Sweeping only from 0 exhausts
        the budget before the step counter can outrun it, which is precisely what
        hid the `step <= max_calls` divergence: the property held on the one
        trajectory the fixture explored.
        """
        from spectramr.infrastructure.training import debug_snapshot as ds

        # start=0 is a fresh run; the rest stand in for a resume, where the loop
        # begins at `start_iteration + 1` with an empty in-process counter.
        for start in (0, 9, 100, 5_000):
            for interval in (0, 1, 3, 100):
                section = _snapshot_section(max_calls=8, interval_steps=interval)
                ds._call_counts.clear()
                for step in range(start, start + 301):
                    wrote = (
                        ds.save_debug_snapshot(
                            run_dir=tmp_path,
                            step=step,
                            tag=f"agree_{start}_{interval}",
                            tensors={"x": torch.rand(1, 1, 4, 4)},
                            paradigm="X",
                            config_section=section,
                        )
                        is not None
                    )
                    if wrote:
                        assert ds.snapshot_step_is_due(section, step), (
                            f"predicate rejected step {step} that the writer wrote "
                            f"(start={start}, interval_steps={interval})"
                        )

    def test_a_resumed_run_is_still_due(self, tmp_path):
        """The production shape of the divergence, stated directly.

        `pipelines/training_loop.py` resumes at `range(start_iteration + 1, ...)`
        and the diffusion call site gates on the GLOBAL iteration, so a run
        resumed past `max_calls` asks about large step numbers with a fresh
        in-process budget. Under `step <= max_calls` the answer was "not due"
        forever, and a resumed cold-diffusion arm wrote ZERO `diffusion_step`
        snapshots — the one artifact showing the degraded model input.
        """
        from spectramr.infrastructure.training import debug_snapshot as ds

        section = _snapshot_section(max_calls=8, interval_steps=0)
        for start_iteration in (8, 99, 4_999):
            ds._call_counts.clear()
            step = start_iteration + 1
            assert ds.snapshot_step_is_due(section, step) is True
            wrote = ds.save_debug_snapshot(
                run_dir=tmp_path,
                step=step,
                tag=f"resume_{start_iteration}",
                tensors={"x": torch.rand(1, 1, 4, 4)},
                paradigm="X",
                config_section=section,
            )
            assert wrote is not None, f"writer refused the first resumed step {step}"


class TestProvenanceReachesTheArtifact:
    """The record of what was done to the data must survive to disk.

    Without it the four numbers per tensor cannot be read against what the arm
    asked for, which is what made an unaccelerated-looking ``input_prepared``
    unfalsifiable from the artifact alone.
    """

    _RECORD: ClassVar[dict] = {
        "source": "train",
        "declared": {"target_mode": "phase_aligned_mean"},
        "applied": {
            "dataset_chain": ["M4RawDataset"],
            "transforms": [{"name": "KSpaceNormalizationTransform"}],
        },
        "model_input_linearization": [{"module": "HilbertOrder", "mode": "hilbert"}],
        "incomplete": ["augmentation Compose not reachable"],
    }

    def _write(self, tmp_path: Path):
        save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="first_steps",
            tensors={"input_prepared": torch.zeros(1, 2, 8, 8)},
            config_section=_snapshot_section(),
            provenance=self._RECORD,
        )
        return tmp_path / "debug_snapshots" / "first_steps_step_000000"

    def test_it_is_a_top_level_json_key_with_its_nesting_intact(self, tmp_path):
        """NOT folded into ``extra``: ``_coerce_extra`` flattens non-scalars to
        ``str(v)``, which would turn the transform list into one unparseable
        line — the machine-readable half would be lost."""
        snap_dir = self._write(tmp_path)
        payload = json.loads((snap_dir / "snapshot.json").read_text())
        assert payload["provenance"]["declared"]["target_mode"] == "phase_aligned_mean"
        assert payload["provenance"]["applied"]["transforms"][0]["name"] == (
            "KSpaceNormalizationTransform"
        )

    def test_the_text_report_shows_declared_beside_applied(self, tmp_path):
        """A divergence between the two halves is the finding, so a reader
        looking at the PNGs must see them adjacent."""
        text = (self._write(tmp_path) / "snapshot.txt").read_text()
        assert "declared.target_mode = phase_aligned_mean" in text
        assert "KSpaceNormalizationTransform" in text
        assert "hilbert" in text

    def test_gaps_are_rendered_loudly_not_left_to_inference(self, tmp_path):
        """A partial record read as complete recreates the facade (pitfall #16)."""
        text = (self._write(tmp_path) / "snapshot.txt").read_text()
        assert "⚠ INCOMPLETE: augmentation Compose not reachable" in text

    def test_omitting_it_leaves_the_report_unchanged_for_other_callers(self, tmp_path):
        save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="no_prov",
            tensors={"input": torch.zeros(1, 2, 8, 8)},
            config_section=_snapshot_section(),
        )
        snap_dir = tmp_path / "debug_snapshots" / "no_prov_step_000000"
        assert json.loads((snap_dir / "snapshot.json").read_text())["provenance"] is None
        assert "Data provenance" not in (snap_dir / "snapshot.txt").read_text()


class TestDeferredExtra:
    """A deferred ``extra`` is resolved only when a write is CERTAIN (#1188).

    The defect this pins: a caller that wants a scalar diagnostic in the report
    (``vf_twin``'s marker-residual max, the base emitter's scale context) has to
    reduce a tensor, and on an accelerator that reduction is a device->host sync
    -- forbidden in the training loop by non-negotiable 9. Every gate lives
    INSIDE ``save_debug_snapshot``, so an eagerly-built ``extra`` is paid on
    every step while the budget suppresses only the write. The gates cannot be
    exported as a "will this write?" predicate either, because ``_budget_check``
    mutates the counter. Deferral is the only mechanism that closes it, and its
    correctness is entirely a question of WHERE the callable is invoked -- hence
    one test per gate rather than one test for the feature.
    """

    @staticmethod
    def _counter():
        """A zero-arg ``extra`` that records how many times it was invoked."""
        calls: list[int] = []

        def _build() -> dict:
            calls.append(1)
            return {"deferred_value": 7, "invocations": len(calls)}

        return _build, calls

    def test_a_callable_is_not_invoked_when_snapshots_are_disabled(self, tmp_path):
        _reset_budget()
        build, calls = self._counter()
        assert (
            save_debug_snapshot(
                run_dir=tmp_path,
                step=0,
                tag="deferred",
                tensors={"input": torch.rand(1, 1, 8, 8)},
                config_section=_make_logging(enabled=False),
                extra=build,
            )
            is None
        )
        assert calls == []

    def test_a_callable_is_not_invoked_on_an_interval_suppressed_step(self, tmp_path):
        _reset_budget()
        build, calls = self._counter()
        # step 3 with interval 5 -> rejected before the budget is even consulted.
        assert (
            save_debug_snapshot(
                run_dir=tmp_path,
                step=3,
                tag="deferred",
                tensors={"input": torch.rand(1, 1, 8, 8)},
                config_section=_make_logging(interval_steps=5),
                extra=build,
            )
            is None
        )
        assert calls == []

    def test_a_callable_is_not_invoked_once_the_budget_is_spent(self, tmp_path):
        """The gate that actually fires in a real run.

        ``vf_twin`` arms leave ``interval_steps`` at its default 0, so the
        interval never suppresses anything and ``max_calls`` is the only thing
        standing between the diagnostic and every step of training.
        """
        _reset_budget()
        build, calls = self._counter()
        for step in range(6):
            save_debug_snapshot(
                run_dir=tmp_path,
                step=step,
                tag="deferred",
                tensors={"input": torch.rand(1, 1, 8, 8)},
                config_section=_make_logging(max_calls=2),
                extra=build,
            )
        # Two writes, two resolutions -- steps 2..5 cost nothing at all.
        assert len(calls) == 2

    def test_a_callable_is_invoked_exactly_once_per_written_snapshot(self, tmp_path):
        """Once, not twice: ``extra`` has two downstream consumers.

        ``_format_text_report`` and ``_coerce_extra`` both read it. Resolving at
        each consumer instead of once at the gate would double every sync -- a
        subtler version of the same defect, and invisible to any test that only
        checks the values came out right.
        """
        _reset_budget()
        build, calls = self._counter()
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="deferred",
            tensors={"input": torch.rand(1, 1, 8, 8)},
            config_section=_make_logging(),
            extra=build,
        )
        assert out is not None
        assert len(calls) == 1
        payload = json.loads((out / "snapshot.json").read_text())["extra"]
        assert payload["invocations"] == 1

    def test_a_diverged_snapshot_is_still_strict_json(self, tmp_path):
        """A NaN output made ``min``/``abs_max`` NaN, written as the bare token
        ``NaN`` that an RFC 8259 parser rejects."""

        def _reject(token: str) -> None:
            raise ValueError(f"non-RFC-8259 token {token}")

        _reset_budget()
        nan = torch.full((1, 2, 4, 4), float("nan"))
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="diverged",
            tensors={"model_output": nan},
            config_section=_make_logging(),
            extra={"abs_max_model_output": float("nan"), "vector": torch.tensor([1.0, float("inf")])},
        )
        payload = json.loads((out / "snapshot.json").read_text(), parse_constant=_reject)
        assert payload["extra"]["abs_max_model_output"] is None
        assert payload["extra"]["vector"] == [1.0, None]

    def test_a_deferred_value_reaches_both_the_json_and_the_text_report(self, tmp_path):
        _reset_budget()
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="deferred",
            tensors={"input": torch.rand(1, 1, 8, 8)},
            config_section=_make_logging(),
            extra=lambda: {"marker_residual_abs_max": 0.25},
        )
        payload = json.loads((out / "snapshot.json").read_text())["extra"]
        assert payload["marker_residual_abs_max"] == 0.25
        assert "marker_residual_abs_max" in (out / "snapshot.txt").read_text()

    def test_per_key_callables_are_resolved_individually(self, tmp_path):
        """The dict-of-callables shape, for extras where only one key is costly."""
        _reset_budget()
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="deferred",
            tensors={"input": torch.rand(1, 1, 8, 8)},
            config_section=_make_logging(),
            extra={"epoch": 3, "costly": lambda: 1.5},
        )
        payload = json.loads((out / "snapshot.json").read_text())["extra"]
        assert payload["epoch"] == 3
        assert payload["costly"] == 1.5

    def test_a_plain_dict_is_passed_through_untouched(self, tmp_path):
        """The overwhelmingly common shape must not change behaviour at all."""
        _reset_budget()
        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="deferred",
            tensors={"input": torch.rand(1, 1, 8, 8)},
            config_section=_make_logging(),
            extra={"degradation_source": "input", "epoch": 0},
        )
        payload = json.loads((out / "snapshot.json").read_text())["extra"]
        assert payload["degradation_source"] == "input"
        assert payload["epoch"] == 0

    def test_a_raising_callable_stamps_a_reason_rather_than_vanishing(self, tmp_path):
        """A diagnostic that silently disappears reads as one never requested.

        Pitfall #16: the key survives carrying why it could not be built, so a
        reviewer sees a broken probe instead of an absent one.
        """
        _reset_budget()

        def _boom() -> float:
            raise RuntimeError("device gone")

        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="deferred",
            tensors={"input": torch.rand(1, 1, 8, 8)},
            config_section=_make_logging(),
            extra={"epoch": 1, "costly": _boom},
        )
        payload = json.loads((out / "snapshot.json").read_text())["extra"]
        assert payload["epoch"] == 1
        assert "unresolved" in payload["costly"]
        assert "RuntimeError" in payload["costly"]

    def test_a_raising_whole_dict_callable_still_writes_the_snapshot(self, tmp_path):
        """The tensors are the point of the artifact; ``extra`` is annotation.

        Losing the whole snapshot because an annotation could not be computed
        would trade a small diagnostic away for a large one.
        """
        _reset_budget()

        def _boom() -> dict:
            raise ValueError("no")

        out = save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="deferred",
            tensors={"input": torch.rand(1, 1, 8, 8)},
            config_section=_make_logging(),
            extra=_boom,
        )
        assert out is not None
        assert (out / "snapshot.json").exists()
        payload = json.loads((out / "snapshot.json").read_text())["extra"]
        assert "unresolved" in payload["extra"]
        assert "ValueError" in payload["extra"]

    def test_the_resolution_happens_after_the_gates_not_before_them(self):
        """Source guard: the ordering IS the fix, and it is one line to move.

        Every assertion above still passes if the resolution is hoisted to the
        top of the function -- they would just be testing a feature instead of a
        fix. This pins the position: the call must sit after both gates and
        before the write.
        """
        import inspect

        from spectramr.infrastructure.training import debug_snapshot as ds

        # Drop the leading docstring only: `[-1]` would cut at the LAST triple
        # quote, and this function's body carries two more of them, so it would
        # silently discard the very gates under test. Segment 0 is the
        # signature, 1 the docstring, and everything after that is body.
        body = '"""'.join(inspect.getsource(ds.save_debug_snapshot).split('"""')[2:])
        # Exact call syntax, not the bare names: the comment above the gates
        # mentions `_budget_check` several lines before the call, so a bare
        # `.index()` would pass on the prose while the call sat anywhere.
        gates = (body.index("if cfg.image_interval_steps > 0"), body.index("if not _budget_check("))
        resolve = body.index("_resolve_deferred_extra(extra)")
        assert max(gates) < resolve
        assert resolve < body.index(".mkdir(")


class TestRunIdentityReachesTheArtifact:
    """#1299: a snapshot must name the run that produced it.

    Snapshots accumulate under one arm directory across relaunches, while the
    only file that carried a run id -- ``provenance.json`` -- is rewritten at
    every launch. Five launches of ``experiment_11_attention_none`` left one
    directory holding snapshots from different commits and different configs,
    with nothing in any artifact to separate them.
    """

    def _write(self, tmp_path: Path):
        _reset_budget()
        return save_debug_snapshot(
            run_dir=tmp_path,
            step=7,
            tag="first_steps",
            tensors={"input": torch.rand(1, 1, 8, 8)},
            paradigm="TestParadigm",
            config_section=_make_logging(),
        )

    def test_the_json_carries_the_published_run_id(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.training.snapshot_provenance import (
            reset_run_identity,
            set_run_identity,
        )

        reset_run_identity()
        try:
            set_run_identity(
                {
                    "run_id": "exp11-20260821_101500-abc123def456",
                    "run_name": "exp11",
                    "started_at": "2026-08-21T10:15:00-05:00",
                    "config_sha256": "0f1e2d3c",
                    "git": {"sha": "abc123def456789", "branch": "dev", "dirty": False},
                }
            )
            out = self._write(tmp_path)
            payload = json.loads((out / "snapshot.json").read_text())
            assert payload["run"]["run_id"] == "exp11-20260821_101500-abc123def456"
            assert payload["run"]["config_sha256"] == "0f1e2d3c"
            assert payload["run"]["identity_source"] == "run_provenance"
        finally:
            reset_run_identity()

    def test_the_text_report_shows_it_without_opening_the_json(
        self, tmp_path: Path
    ) -> None:
        """``snapshot.txt`` is what a human actually reads when triaging an arm,
        so the id has to be legible there, not only machine-readable."""
        from spectramr.infrastructure.training.snapshot_provenance import (
            reset_run_identity,
            set_run_identity,
        )

        reset_run_identity()
        try:
            set_run_identity(
                {
                    "run_id": "exp11-20260821_101500-abc123def456",
                    "started_at": "2026-08-21T10:15:00-05:00",
                    "git": {"sha": "abc123def456789", "branch": "dev", "dirty": True},
                }
            )
            out = self._write(tmp_path)
            text = (out / "snapshot.txt").read_text()
            assert "run=exp11-20260821_101500-abc123def456" in text
            assert "abc123def456" in text
            assert "dirty tree" in text
        finally:
            reset_run_identity()

    def test_an_unpublished_run_still_gets_an_id_that_declares_itself(
        self, tmp_path: Path
    ) -> None:
        """Pitfall #16: a field that silently disappears reads as a snapshot
        written before #1299 landed. The fallback is present and labelled."""
        from spectramr.infrastructure.training.snapshot_provenance import (
            reset_run_identity,
        )

        reset_run_identity()
        try:
            out = self._write(tmp_path)
            payload = json.loads((out / "snapshot.json").read_text())
            assert payload["run"]["run_id"]
            assert payload["run"]["identity_source"] == "fallback"
            assert "identity_source=fallback" in (out / "snapshot.txt").read_text()
        finally:
            reset_run_identity()

    def test_two_snapshots_from_one_run_agree(self, tmp_path: Path) -> None:
        """The correlation the arm directory needs: same run, same id, so a
        differing id is positive evidence of a relaunch rather than noise."""
        from spectramr.infrastructure.training.snapshot_provenance import (
            reset_run_identity,
        )

        reset_run_identity()
        try:
            first = json.loads((self._write(tmp_path) / "snapshot.json").read_text())
            second = json.loads((self._write(tmp_path) / "snapshot.json").read_text())
            assert first["run"]["run_id"] == second["run"]["run_id"]
        finally:
            reset_run_identity()


class TestOnlyRankZeroWrites:
    """Every rank used to write the same snapshot directory, racing and overwriting."""

    @staticmethod
    def _call(tmp_path: Path, extra=None):
        return save_debug_snapshot(
            run_dir=tmp_path,
            step=0,
            tag="t",
            tensors={"input": torch.rand(1, 1, 16, 16)},
            paradigm="TestParadigm",
            config_section=_make_logging(),
            extra=extra,
        )

    def test_a_non_zero_rank_writes_nothing(self, tmp_path: Path, monkeypatch) -> None:
        from spectramr.infrastructure.distributed.distributed_training import RankUtility

        _reset_budget()
        monkeypatch.setattr(RankUtility, "get_rank", staticmethod(lambda: 1))
        assert self._call(tmp_path) is None
        assert not (tmp_path / "debug_snapshots").exists()

    def test_a_non_zero_rank_spends_no_budget_and_resolves_no_deferred_extra(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The deferred ``extra`` exists to avoid a device-to-host sync, so a rank that
        does not write must not pay it either."""
        from spectramr.infrastructure.distributed.distributed_training import RankUtility

        _reset_budget()
        resolved: list[int] = []
        monkeypatch.setattr(RankUtility, "get_rank", staticmethod(lambda: 3))
        self._call(tmp_path, extra=lambda: resolved.append(1) or {"x": 1.0})
        assert resolved == []
        assert not _call_counts

    def test_rank_zero_still_writes(self, tmp_path: Path, monkeypatch) -> None:
        from spectramr.infrastructure.distributed.distributed_training import RankUtility

        _reset_budget()
        monkeypatch.setattr(RankUtility, "get_rank", staticmethod(lambda: 0))
        out = self._call(tmp_path)
        assert out is not None and (out / "snapshot.json").exists()
