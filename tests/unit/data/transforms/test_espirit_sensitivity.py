"""Coil maps have to be produced by something, and be complex when they arrive.

`CoilCombineTransform` consumes a ``sensitivity`` key and the subject builder
can load maps from a manifest ``sensitivity_path``, but nothing computed them.
An arm whose objective needs the coil geometry had no route to it short of
precomputing files offline.

The half that is easy to get silently wrong is the dtype. The subject builder's
second branch stores ``sensitivity`` as ``sens_tensor.abs()``, and a magnitude
map is not merely lossy for a phase-sensitive consumer: ``I - s s^H`` with real
``s`` is a real projector, so it computes a different quantity with no error
anywhere. These tests pin that the complex map exists, that it is complex, and
that the magnitude spelling does not shadow it.
"""

from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.data.transforms.espirit_sensitivity import ESPIRiTSensitivityTransform
from spectramr.infrastructure.physics.fft_ops import fft2c

_C, _S = 4, 64


def _subject() -> tio.Subject:
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _S), torch.linspace(-1, 1, _S), indexing="ij")
    anatomy = ((xx**2 + yy**2) < 0.6).to(torch.complex64)
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = torch.stack([torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres]).to(
        torch.complex64
    )
    sens = sens / sens.abs().pow(2).sum(0, keepdim=True).sqrt().clamp(min=1e-6)
    kspace = fft2c(sens * anatomy)
    interleaved = torch.zeros(2 * _C, _S, _S, 1)
    interleaved[0::2, ..., 0] = kspace.real
    interleaved[1::2, ..., 0] = kspace.imag
    return tio.Subject(kspace=tio.ScalarImage(tensor=interleaved))


class TestItProducesWhatConsumersNeed:
    def test_it_emits_both_spellings(self) -> None:
        out = ESPIRiTSensitivityTransform(acs_size=24)(_subject())
        assert "sensitivity_complex" in out
        assert "sensitivity" in out

    def test_the_complex_map_is_complex(self) -> None:
        """The whole point. A real map silently changes what a projector computes."""
        out = ESPIRiTSensitivityTransform(acs_size=24)(_subject())
        assert torch.is_complex(out["sensitivity_complex"].data)

    def test_the_magnitude_spelling_does_not_shadow_it(self) -> None:
        """Both keys, so a SENSE combine and a phase consumer each get theirs."""
        out = ESPIRiTSensitivityTransform(acs_size=24)(_subject())
        assert not torch.is_complex(out["sensitivity"].data)
        assert torch.allclose(out["sensitivity"].data, out["sensitivity_complex"].data.abs())

    def test_the_maps_are_unit_norm_inside_the_support(self) -> None:
        """ESPIRiT's eigenvectors are orthonormal; the soft weight tapers outside."""
        out = ESPIRiTSensitivityTransform(acs_size=24)(_subject())
        norm = out["sensitivity_complex"].data[..., 0].abs().pow(2).sum(0).sqrt()
        assert float(norm.max()) == pytest.approx(1.0, abs=0.02)
        assert float(norm.min()) == pytest.approx(0.0, abs=1e-5)

    def test_the_shape_matches_the_coil_count(self) -> None:
        out = ESPIRiTSensitivityTransform(acs_size=24)(_subject())
        assert out["sensitivity_complex"].data.shape == (_C, _S, _S, 1)


class TestItFailsLoudly:
    def test_no_kspace_raises_by_default(self) -> None:
        """Returning the subject untouched leaves a consumer's maps absent silently."""
        subject = tio.Subject(image=tio.ScalarImage(tensor=torch.randn(1, _S, _S, 1)))
        with pytest.raises(ValueError, match="no k-space"):
            ESPIRiTSensitivityTransform()(subject)

    def test_strict_false_is_an_explicit_opt_out(self) -> None:
        subject = tio.Subject(image=tio.ScalarImage(tensor=torch.randn(1, _S, _S, 1)))
        out = ESPIRiTSensitivityTransform(strict=False)(subject)
        assert "sensitivity_complex" not in out

    def test_combined_coils_raise(self) -> None:
        """An odd channel count means there is no array left to calibrate."""
        subject = tio.Subject(kspace=tio.ScalarImage(tensor=torch.randn(1, _S, _S, 1)))
        with pytest.raises(ValueError, match="already combined"):
            ESPIRiTSensitivityTransform()(subject)

    def test_an_undersized_acs_raises_rather_than_returning_a_bad_map(self) -> None:
        """`estimate_csm_espirit` names the minimum for the coil count and kernel."""
        with pytest.raises(ValueError, match="rank-deficient"):
            ESPIRiTSensitivityTransform(acs_size=8)(_subject())


class TestItIsReachableFromYaml:
    def test_it_is_registered(self) -> None:
        import subprocess
        import sys

        probe = (
            "import spectramr.data.transforms;"
            "from spectramr.data.transforms.registry import TRANSFORM_REGISTRY;"
            "import sys;"
            "sys.exit(0 if 'espirit_sensitivity' in TRANSFORM_REGISTRY else 1)"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, check=False)
        assert result.returncode == 0, (
            "espirit_sensitivity is not registered by importing "
            "spectramr.data.transforms alone, so no YAML can select it.\n"
            + result.stderr.decode()[-500:]
        )

    def test_it_declares_what_it_produces(self) -> None:
        """The registry's `produces` is what a reader checks before wiring an arm."""
        import spectramr.data.transforms  # noqa: F401
        from spectramr.data.transforms.registry import TRANSFORM_REGISTRY

        entry = TRANSFORM_REGISTRY["espirit_sensitivity"]
        assert set(entry.produces) == {"sensitivity", "sensitivity_complex"}


def test_the_mixin_prefers_the_complex_map() -> None:
    """End of the chain: a magnitude map reaching a phase consumer is the defect."""
    from spectramr.infrastructure.training.strategies.mixins.reconstruction import (
        ReconstructionMixin,
    )

    class _Probe(ReconstructionMixin):
        def __init__(self) -> None:
            self.env = None

        @property
        def state(self):
            class _Model:
                input_type = "kspace"

            class _Config:
                model = _Model()

            class _State:
                config = _Config()

            return _State()

    out = ESPIRiTSensitivityTransform(acs_size=24)(_subject())
    maps = out["sensitivity_complex"].data[..., 0].unsqueeze(0)
    raw = {"sensitivity_complex": maps, "sensitivity": maps.abs()}
    context = _Probe()._prepare_batch_context_reconstruction(
        torch.randn(1, 2 * _C, _S, _S), torch.randn(1, 2 * _C, _S, _S), batch=raw
    )
    picked = context.get("coil_sensitivities")
    assert picked is not None
    assert torch.is_complex(picked), "the magnitude spelling shadowed the complex one"


def _multislice_subject(depth: int) -> tio.Subject:
    """A volume whose slices carry DIFFERENT anatomy under one coil array."""
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _S), torch.linspace(-1, 1, _S), indexing="ij")
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = torch.stack([torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres]).to(
        torch.complex64
    )
    sens = sens / sens.abs().pow(2).sum(0, keepdim=True).sqrt().clamp(min=1e-6)
    interleaved = torch.zeros(2 * _C, _S, _S, depth)
    for d in range(depth):
        anatomy = ((xx**2 + yy**2) < 0.25 + 0.08 * d).to(torch.complex64)
        kspace = fft2c(sens * anatomy)
        interleaved[0::2, ..., d] = kspace.real
        interleaved[1::2, ..., d] = kspace.imag
    return tio.Subject(kspace=tio.ScalarImage(tensor=interleaved))


class TestEverySliceIsCalibrated:
    """A depth-1 map on a depth-D subject is not merely coarse -- it is fatal.

    TorchIO's sampler calls ``check_consistent_space`` before drawing a patch
    and raises when two images in a subject disagree about ``spatial_shape``, so
    a map that does not carry the source's depth crashes the loader rather than
    degrading. Coil sensitivities also genuinely vary along the slice axis, so
    broadcasting one slice's map over the volume would be wrong even if TorchIO
    tolerated it.
    """

    @pytest.mark.parametrize("depth", [1, 5])
    def test_the_maps_carry_the_source_depth(self, depth: int) -> None:
        out = ESPIRiTSensitivityTransform(acs_size=24)(_multislice_subject(depth))
        assert out["sensitivity_complex"].data.shape == (_C, _S, _S, depth)
        assert out["sensitivity"].data.shape == (_C, _S, _S, depth)

    def test_a_patch_sampler_accepts_the_subject(self) -> None:
        """The production path: `patch_size` is `[256, 256, 1]` on every n2n arm."""
        out = ESPIRiTSensitivityTransform(acs_size=24)(_multislice_subject(5))
        sampler = tio.data.UniformSampler(patch_size=(_S, _S, 1))
        patch = next(iter(sampler(out, num_patches=1)))
        assert patch["sensitivity_complex"].data.shape == (_C, _S, _S, 1)
        assert patch["kspace"].data.shape == (2 * _C, _S, _S, 1)

    def test_slices_get_their_own_maps(self) -> None:
        """Not the middle slice broadcast: differing anatomy gives differing maps."""
        out = ESPIRiTSensitivityTransform(acs_size=24)(_multislice_subject(5))
        maps = out["sensitivity_complex"].data
        assert not torch.allclose(maps[..., 0], maps[..., 4], atol=1e-3)

    def test_it_agrees_with_calibrating_one_slice_alone(self) -> None:
        """Batching over depth is an optimisation, not a different answer."""
        volume = _multislice_subject(5)
        batched = ESPIRiTSensitivityTransform(acs_size=24)(volume)["sensitivity_complex"].data
        alone = tio.Subject(
            kspace=tio.ScalarImage(tensor=volume["kspace"].data[..., 2:3].clone())
        )
        single = ESPIRiTSensitivityTransform(acs_size=24)(alone)["sensitivity_complex"].data
        assert torch.allclose(batched[..., 2:3], single, atol=1e-5)
