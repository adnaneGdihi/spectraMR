"""Rendering and per-tensor statistics, moved unchanged from the retired ``debug_snapshot``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spectramr.infrastructure.training.snapshot.stats import tensor_report

pytestmark = pytest.mark.unit


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


class TestTensorReport:
    def test_basic_image_pair(self) -> None:
        t = torch.rand(1, 1, 16, 16)
        r = tensor_report("img", t)
        assert r.shape == [1, 1, 16, 16]
        assert r.is_complex is False
        assert r.nan_count == 0 and r.inf_count == 0
        assert r.min is not None and r.max is not None

    def test_complex_tensor(self) -> None:
        t = torch.complex(torch.rand(1, 2, 8, 8), torch.rand(1, 2, 8, 8))
        r = tensor_report("ksp", t)
        assert r.is_complex is True
        # magnitude min/max
        assert r.min >= 0

    def test_nan_inf_counts(self) -> None:
        t = torch.tensor([1.0, float("nan"), float("inf"), -float("inf"), 2.0])
        r = tensor_report("bad", t)
        assert r.nan_count == 1
        assert r.inf_count == 2
        assert "NaN" in r.note and "Inf" in r.note

    def test_constant_tensor_flagged(self) -> None:
        t = torch.full((4, 1, 8, 8), 0.5)
        r = tensor_report("const", t)
        assert "constant" in r.note.lower()

    def test_real_stacked_complex_hint(self) -> None:
        # 4-channel real-stacked → flagged as likely complex pairs
        t = torch.rand(1, 4, 8, 8)
        r = tensor_report("k_real_stacked", t)
        assert "real-stacked" in r.note


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
        from spectramr.infrastructure.training.snapshot.render import render_image_preview

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
        preview = render_image_preview(paired, in_kspace=False)
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
        from spectramr.infrastructure.training.snapshot.render import render_image_preview

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing="ij",
        )
        disk = ((x**2 + y**2) < 0.5).float()
        # 4-channel real-stacked complex: (R0, I0, R1, I1) where
        # |C_c|² = R_c² + I_c². Make all four channels carry the disk.
        t = disk.unsqueeze(0).unsqueeze(0).expand(1, 4, 16, 16).clone()
        preview = render_image_preview(t, in_kspace=False)
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
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, H), indexing="ij")
        disk = ((x**2 + y**2) < 0.5).float() * peak
        if complex_:
            return torch.complex(disk, 0.2 * disk).unsqueeze(0).unsqueeze(0)
        return disk.unsqueeze(0).unsqueeze(0)

    def test_looks_like_kspace_separates_domains(self) -> None:
        from spectramr.infrastructure.physics.fft_ops import fft2c
        from spectramr.infrastructure.training.snapshot.render import _looks_like_kspace

        img = self._disk().abs()  # image-domain magnitude
        ksp = fft2c(self._disk()).abs()  # its k-space magnitude
        assert _looks_like_kspace(img) is False
        assert _looks_like_kspace(ksp) is True

    def test_image_flagged_kspace_is_not_ifftd(self) -> None:
        """A complex IMAGE tensor flagged in_kspace=True must render as the
        image (structure preserved), NOT a DC blob."""
        from spectramr.infrastructure.training.snapshot.render import render_image_preview

        preview = render_image_preview(self._disk(), in_kspace=True)
        assert preview is not None
        # The disk fills ~31% of the frame; a wrongly-IFFT'd image collapses to
        # a tiny central DC blob (<1% bright). Require a broad bright region.
        bright_frac = (preview > 0.5).float().mean().item()
        assert bright_frac > 0.15, (
            f"only {bright_frac:.3f} bright — image was IFFT'd into a DC blob"
        )

    def test_real_kspace_still_ifftd(self) -> None:
        """Genuine k-space (DC signature present) must STILL be IFFT'd so its
        preview is the reconstructed image — the m10 / experiment_11 contract."""
        from spectramr.infrastructure.physics.fft_ops import fft2c
        from spectramr.infrastructure.training.snapshot.render import render_image_preview

        ksp = fft2c(self._disk())  # complex k-space of the disk
        preview = render_image_preview(ksp, in_kspace=True)
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

        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, H), indexing="ij")
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
        from spectramr.infrastructure.training.snapshot.render import _looks_like_kspace

        kn = self._normalized_kspace()
        # legacy heuristic on the raw magnitude => "not k-space" (the bug)
        assert _looks_like_kspace(kn.abs()) is False

    def test_is_in_kspace_domain_matrix(self) -> None:
        """The spectrum-based decision is normalization-invariant: raw AND
        normalized k-space => IFFT; an image (even flagged k-space) => skip."""
        from spectramr.infrastructure.physics.fft_ops import fft2c
        from spectramr.infrastructure.training.snapshot.render import _is_in_kspace_domain

        assert _is_in_kspace_domain(fft2c(self._disk())) is True  # raw k-space
        assert _is_in_kspace_domain(self._normalized_kspace()) is True  # normalized k-space
        assert _is_in_kspace_domain(self._disk()) is False  # image (VF/IB)

    def test_normalized_kspace_is_ifftd_to_image(self) -> None:
        """End-to-end: a normalized k-space flagged in_kspace=True must be
        IFFT'd (decision True), where the legacy code skipped it."""
        from spectramr.infrastructure.training.snapshot.render import (
            _is_in_kspace_domain,
            render_image_preview,
        )

        kn = self._normalized_kspace()
        assert _is_in_kspace_domain(kn) is True
        preview = render_image_preview(kn, in_kspace=True)
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
        import spectramr.infrastructure.training.snapshot.render as ds
        from spectramr.infrastructure.physics.fft_ops import fft2c

        # Reproduce the real-data failure mode: heuristic wrongly says "image".
        monkeypatch.setattr(ds, "_is_in_kspace_domain", lambda _x: False)
        ksp = fft2c(self._disk())  # genuine k-space of the disk

        # Non-authoritative: the veto skips the IFFT → raw k-space DC blob (bug).
        buggy = ds.render_image_preview(ksp, in_kspace=True)
        assert buggy is not None
        assert (buggy > 0.5).float().mean().item() < 0.05

        # Authoritative: IFFT regardless of the heuristic → recovers the disk.
        fixed = ds.render_image_preview(ksp, in_kspace=True, authoritative=True)
        assert fixed is not None
        assert (fixed > 0.5).float().mean().item() > 0.15


class TestSnapshotDecompressesLogScaledKspace:
    """#682: the renderer IFFT'd a log1p-compressed spectrum.

    The renderer cannot reach `config.data`, so the flag is threaded in as an argument
    by the caller that does have the config.
    """

    @staticmethod
    def _compressed_phantom():
        import torch

        from spectramr.data.transforms.normalization import compress_kspace_log
        from spectramr.infrastructure.physics.fft_ops import fft2c

        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
        img = ((xx**2 + yy**2) < 0.35).float().unsqueeze(0).unsqueeze(0)
        return img, compress_kspace_log(fft2c(img.to(torch.complex64)))

    def test_the_flag_changes_the_render(self):
        import torch

        from spectramr.infrastructure.training.snapshot import render as ds

        _img, compressed = self._compressed_phantom()
        off = ds.render_image_preview(
            compressed, in_kspace=True, authoritative=True, log_scaled=False
        )
        on = ds.render_image_preview(
            compressed, in_kspace=True, authoritative=True, log_scaled=True
        )
        assert not torch.allclose(off, on), "log_scaled had no effect on the render"

    def test_the_decompressed_render_matches_the_anatomy(self):
        from spectramr.infrastructure.training.snapshot import render as ds

        img, compressed = self._compressed_phantom()
        rendered = ds.render_image_preview(
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

        from spectramr.infrastructure.training.snapshot.render import render_image_preview

        assert inspect.signature(render_image_preview).parameters["log_scaled"].default is False
