"""One decompression step for every k-space visualization path (#682).

`data.processing.enable_log_scaling` compresses k-space with `m -> log1p(m)`. The
inverse had exactly ONE caller in the tree -- inside the diffusion strategy, gated
on `is_cold_diffusion` -- so every other arm inverse-FFT'd the *compressed*
spectrum.

That is not a scaled image. `log1p` is a per-bin nonlinearity, so it reweights the
spectrum before the transform. These tests measure the consequence rather than
asserting a call happened: the render is only ~0.78 correlated with the truth
without decompression, and exactly 1.0 with it.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from spectramr.data.transforms.normalization import (  # noqa: E402
    compress_kspace_log,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c  # noqa: E402
from spectramr.infrastructure.training.utils.kspace_view import (  # noqa: E402
    decompress_for_view,
    log_scaling_enabled,
)


def _phantom(n: int = 64) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij"
    )
    return ((xx**2 + yy**2) < 0.35).float().unsqueeze(0).unsqueeze(0)


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = (a - a.mean()).flatten()
    b = (b - b.mean()).flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


class TestDecompressionRestoresTheImage:
    def test_ifft_of_compressed_kspace_is_not_the_image(self):
        """The premise. If this ever passes, the defect is gone by other means."""
        img = _phantom()
        compressed = compress_kspace_log(fft2c(img.to(torch.complex64)))

        wrong = ifft2c(compressed)[0, 0].abs()

        assert _corr(wrong, img[0, 0]) < 0.95, (
            "IFFT of a log1p-compressed spectrum should NOT reproduce the image; "
            "if it does, this fixture no longer exercises #682"
        )

    def test_decompressing_first_restores_it_exactly(self):
        img = _phantom()
        compressed = compress_kspace_log(fft2c(img.to(torch.complex64)))

        right = ifft2c(decompress_for_view(compressed, log_scaled=True))[0, 0].abs()

        assert _corr(right, img[0, 0]) == pytest.approx(1.0, abs=1e-3)

    def test_the_physical_scale_comes_back(self):
        """expm1 must INCREASE |k|max -- the tell that the inverse actually fired."""
        img = _phantom()
        k = fft2c(img.to(torch.complex64))
        compressed = compress_kspace_log(k)

        restored = decompress_for_view(compressed, log_scaled=True)

        assert float(compressed.abs().max()) < float(k.abs().max())
        assert float(restored.abs().max()) == pytest.approx(
            float(k.abs().max()), rel=1e-3
        )


class TestItIsANoOpWhenNotCompressed:
    def test_flag_false_returns_the_input_untouched(self):
        """Call sites invoke it unconditionally, so this must be free and exact."""
        k = fft2c(_phantom().to(torch.complex64))
        assert torch.equal(decompress_for_view(k, log_scaled=False), k)

    def test_a_non_kspace_tensor_is_left_alone_even_when_the_flag_is_on(self):
        """An odd-channel real tensor was never compressed; expm1 would wreck it."""
        x = torch.rand(1, 3, 8, 8)
        assert torch.equal(decompress_for_view(x, log_scaled=True), x)


def _cfg(*, normalize: bool, log_scaling: bool):
    from spectramr.config.schemas.data import DataProcessingConfigSchema

    processing = DataProcessingConfigSchema(
        enable_kspace_normalization=normalize, enable_log_scaling=log_scaling
    )
    return types.SimpleNamespace(data=types.SimpleNamespace(processing=processing))


class TestFlagResolution:
    @pytest.mark.parametrize("enabled", [True, False])
    def test_reads_the_nested_ssot_path(self, enabled):
        assert log_scaling_enabled(_cfg(normalize=True, log_scaling=enabled)) is enabled

    def test_log_scaling_without_normalization_is_not_compressed(self):
        """#2353: the compression runs inside the k-space normalization, so
        ``enable_log_scaling`` alone leaves k-space linear. Answering True here
        ``expm1``s a spectrum that was never compressed."""
        assert log_scaling_enabled(_cfg(normalize=False, log_scaling=True)) is False

    def test_absent_block_defaults_to_false(self):
        """False is the safe direction: render what you were given, rather than
        expm1 a tensor that was never compressed."""
        assert log_scaling_enabled(types.SimpleNamespace()) is False
        assert log_scaling_enabled(types.SimpleNamespace(data=None)) is False


class TestARenamedFlagCannotGoQuiet:
    """`getattr(processing, "enable_log_scaling", False)` was rename-blind.

    A declared field read through a defaulting `getattr` is how a mechanism gets
    disabled in silence: rename the field, the guard keeps the old spelling,
    every call returns False, #682 comes back, and nothing goes red because
    "absent" and "off" are the same boolean. This repo has already lost eight
    mechanisms that way. An absent BLOCK and an absent FIELD are different facts.
    """

    def test_a_processing_block_missing_the_field_raises(self):
        cfg = types.SimpleNamespace(
            data=types.SimpleNamespace(processing=types.SimpleNamespace(other=1))
        )
        with pytest.raises(AttributeError, match="enable_log_scaling"):
            log_scaling_enabled(cfg)

    def test_the_absent_block_path_still_returns_false(self):
        """Anti-vacuity: the raise must not swallow the legitimate stub case."""
        assert log_scaling_enabled(types.SimpleNamespace()) is False
        assert log_scaling_enabled(types.SimpleNamespace(data=None)) is False
        assert (
            log_scaling_enabled(types.SimpleNamespace(data=types.SimpleNamespace()))
            is False
        )

    def test_the_real_schema_still_resolves(self):
        """The guard must not reject the field as it is actually declared."""
        from spectramr.config.schemas.data import DataProcessingConfigSchema

        assert "enable_log_scaling" in DataProcessingConfigSchema.model_fields


class TestANoOpDecompressionIsReported:
    """The two host transfers used to feed a debug line and nothing else.

    The comment stated the tell ("expm1 must INCREASE |k|max") while the code
    never tested it, so at the default log level the syncs bought nothing. A
    decompression that changes nothing means the render is still a
    compressed-domain artifact, which is #682 wearing the fix's own clothes.
    """

    def test_a_no_op_inverse_warns(self, caplog, monkeypatch):
        import logging

        from spectramr.infrastructure.training.utils import kspace_view

        monkeypatch.setattr(
            "spectramr.data.transforms.normalization.decompress_kspace_log",
            lambda x, channel_dim=1: x,
        )
        x = torch.rand(1, 2, 8, 8) + 0.5

        with caplog.at_level(logging.WARNING):
            kspace_view.decompress_for_view(x, log_scaled=True)

        assert any("did not expand" in r.getMessage() for r in caplog.records)

    def test_a_real_inverse_does_not_warn(self, caplog):
        """Anti-vacuity: the guard must stay quiet on the working path."""
        import logging

        from spectramr.infrastructure.training.utils import kspace_view

        x = torch.rand(1, 2, 8, 8) + 0.5
        with caplog.at_level(logging.WARNING):
            out = kspace_view.decompress_for_view(x, log_scaled=True)

        assert float(out.abs().max()) > float(x.abs().max())
        assert not [r for r in caplog.records if "did not expand" in r.getMessage()]


def _kspace(peak: float = 2407.0, n: int = 64) -> torch.Tensor:
    """Complex k-space of the phantom, rescaled so ``|k|max == peak``.

    ``2407.0`` is the abs-max ``experiment_11_attention_none``'s snapshot
    actually recorded, and is itself the proof that tensor was uncompressed:
    ``log1p`` saturates at ``ln(FLT_MAX) ~ 88.7`` in float32.
    """
    k = fft2c(_phantom(n).to(torch.complex64))
    return k * (peak / float(k.abs().max()))


class TestASpuriousDecompressionIsRefused:
    """The other direction of the guard, which used to pass in silence.

    ``TestANoOpDecompressionIsReported`` covers decompression that fails to
    expand ``|k|max``. Spurious ``expm1`` -- applied to a tensor that was never
    compressed -- always *expands*, so it satisfied that check and warned about
    nothing while producing the worse render: every coefficient above
    ``DECOMPRESS_MAGNITUDE_CEILING`` collapses to ``expm1(30) ~ 1e13``, phase
    survives, and the IFFT draws a phase-only edge map.

    ``experiment_11_attention_none`` declared ``enable_log_scaling: true`` and
    reached ``train_step`` with an uncompressed batch, so the declaration and the
    tensor disagreed for the whole run with nothing in the log.
    """

    def test_a_tensor_above_the_ceiling_is_returned_untouched(self, caplog):
        import logging

        from spectramr.data.transforms.normalization import (
            DECOMPRESS_MAGNITUDE_CEILING,
        )

        physical = _kspace()
        assert float(physical.abs().max()) > DECOMPRESS_MAGNITUDE_CEILING

        with caplog.at_level(logging.WARNING):
            out = decompress_for_view(physical, log_scaled=True)

        assert out is physical, "the tensor was modified despite the refusal"
        assert any("refusing to decompress" in r.getMessage() for r in caplog.records)

    # The render-fidelity half of this guard is pinned end-to-end, through the
    # real `_render_image_preview` path, in
    # tests/unit/infrastructure/training/test_debug_snapshot_log_scaled_keys.py
    # (`test_render_refuses_to_decompress_what_cannot_be_compressed`). Repeating
    # it here against a hand-rolled IFFT would only add a second baseline to keep
    # in sync.

    def test_the_warning_names_both_diagnoses(self, caplog):
        """|k|max > ceiling has two causes, and they need different fixes.

        For a pipeline tensor it proves the tensor was never compressed. For a
        ``model_output*`` tensor it may instead mean the prediction diverged in
        compressed units -- the ceiling constant's own docstring records this
        arm's kernelized-attention sibling reaching ~1750 at iter 1000. Claiming
        only the first would send a diverged run after a config bug.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            decompress_for_view(_kspace(), log_scaled=True)

        message = "\n".join(r.getMessage() for r in caplog.records)
        assert "never compressed" in message
        assert "DIVERGED" in message

    def test_a_genuinely_compressed_tensor_is_still_decompressed(self, caplog):
        """Anti-vacuity: the guard must not become a blanket skip."""
        import logging

        compressed = compress_kspace_log(_kspace())

        with caplog.at_level(logging.WARNING):
            out = decompress_for_view(compressed, log_scaled=True)

        assert float(out.abs().max()) > float(compressed.abs().max())
        assert not [
            r for r in caplog.records if "refusing to decompress" in r.getMessage()
        ]

    def test_the_round_trip_survives_the_guard(self):
        """compress -> decompress_for_view must still recover the spectrum."""
        physical = _kspace()
        recovered = decompress_for_view(
            compress_kspace_log(physical), log_scaled=True
        )
        assert torch.allclose(
            recovered, physical, rtol=1e-2, atol=1e-2 * float(physical.abs().max())
        )


class TestDecompressScoredAgainst:
    """One decision, taken on the reference, applied to every scored tensor."""

    @staticmethod
    def _compressed(seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.log1p(torch.rand(1, 4, 8, 8, generator=g) * 50.0)

    def test_a_decompressed_reference_decompresses_every_other_tensor(self) -> None:
        from spectramr.infrastructure.training.utils.kspace_view import (
            decompress_scored_against,
        )

        ref, pred = self._compressed(0), self._compressed(1)
        pred[..., 0, 0] = 40.0  # past the ceiling: judged alone it would stay compressed
        out_ref, out = decompress_scored_against(
            ref, {"pred": pred, "none": None}, log_scaled=True
        )
        from spectramr.data.transforms.normalization import decompress_kspace_log

        torch.testing.assert_close(out_ref, decompress_kspace_log(ref, channel_dim=1))
        torch.testing.assert_close(out["pred"], decompress_kspace_log(pred, channel_dim=1))
        assert out["none"] is None

    @pytest.mark.parametrize("log_scaled", [False, True])
    def test_a_reference_left_as_is_leaves_every_other_tensor_as_is(self, log_scaled) -> None:
        """Not log-scaled, or a reference that was never compressed: nothing moves."""
        from spectramr.infrastructure.training.utils.kspace_view import (
            decompress_scored_against,
        )

        ref = self._compressed(2) if not log_scaled else torch.rand(1, 4, 8, 8) * 100.0
        pred = self._compressed(3)
        out_ref, out = decompress_scored_against(ref, {"pred": pred}, log_scaled=log_scaled)
        assert out_ref is ref and out["pred"] is pred
