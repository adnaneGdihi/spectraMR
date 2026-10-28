"""Tests for ``SliceProfileTransform``.

Targets ``spectramr.data.transforms.slice_profile``. Physically realistic
slice-profile blur + decimation along the through-plane axis, with
three profile shapes (apodised sinc, Gauss, Shinnar-Le Roux).

Categories:

- Construction: invalid ``profile`` / ``slice_thickness_mm`` /
  ``kernel_length`` raise ``ValueError``
- Kernel functions: each profile returns a normalised 1-D kernel
- ``_convolve_along_axis``: identity-kernel convolution preserves data;
  blur kernel reduces high-frequency energy
- Forward: produces same-shape output when ``decimate=True``
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torchio as tio

from spectramr.data.transforms.slice_profile import (
    SliceProfileTransform,
    _convolve_along_axis,
)
from spectramr.infrastructure.physics.degradations.blocks.slice_physics import (
    _gauss_kernel,
    _sinc_gauss_kernel,
    _slr_kernel,
)

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_unknown_profile_raises() -> None:
    """Invalid profile name → ``ValueError``."""
    with pytest.raises(ValueError, match="profile must be one of"):
        SliceProfileTransform(profile="cosine")


def test_non_positive_slice_thickness_raises() -> None:
    """``slice_thickness_mm <= 0`` → ``ValueError``."""
    with pytest.raises(ValueError, match="slice_thickness_mm"):
        SliceProfileTransform(slice_thickness_mm=0.0)


def test_short_kernel_raises() -> None:
    """``kernel_length < 3`` → ``ValueError``."""
    with pytest.raises(ValueError, match="kernel_length"):
        SliceProfileTransform(kernel_length=2)


def test_slr_overrides_kernel_length() -> None:
    """``shinnar_le_roux`` profile pins kernel length to 33."""
    transform = SliceProfileTransform(
        profile="shinnar_le_roux", kernel_length=99
    )
    assert transform.kernel_length == 33


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------


def test_gauss_kernel_normalised() -> None:
    """Gaussian kernel sums to ≈ 1."""
    k = _gauss_kernel(fwhm_voxels=4.0, length=21)
    assert pytest.approx(k.sum(), abs=1e-6) == 1.0


def test_sinc_gauss_kernel_normalised() -> None:
    """Apodised-sinc kernel sums to ≈ 1 after sign-correction."""
    k = _sinc_gauss_kernel(fwhm_voxels=4.0, length=21)
    assert pytest.approx(k.sum(), abs=1e-6) == 1.0


def test_slr_kernel_length_is_33() -> None:
    """SLR kernel is exactly 33 taps long."""
    k = _slr_kernel()
    assert k.shape == (33,)


# ---------------------------------------------------------------------------
# 1-D convolution
# ---------------------------------------------------------------------------


def test_identity_kernel_preserves_data() -> None:
    """A delta kernel convolves to (approximately) the input."""
    data = torch.randn(1, 8, 8, 8)
    delta = np.zeros(5)
    delta[2] = 1.0
    out = _convolve_along_axis(data, delta, axis=3)
    assert torch.allclose(out, data, atol=1e-5)


def test_smoothing_kernel_reduces_hf_energy() -> None:
    """A Gaussian kernel reduces along-axis high-frequency energy."""
    torch.manual_seed(0)
    data = torch.randn(1, 8, 8, 16)
    kernel = _gauss_kernel(fwhm_voxels=4.0, length=11)
    smoothed = _convolve_along_axis(data, kernel, axis=3)
    hf_orig = data.diff(dim=3).abs().sum().item()
    hf_smooth = smoothed.diff(dim=3).abs().sum().item()
    assert hf_smooth < hf_orig


def test_axis_out_of_bounds_raises() -> None:
    """Negative or too-large axis → ``ValueError``."""
    data = torch.randn(1, 4, 4, 4)
    kernel = np.ones(3) / 3.0
    with pytest.raises(ValueError, match="axis"):
        _convolve_along_axis(data, kernel, axis=10)


# ---------------------------------------------------------------------------
# End-to-end TorchIO transform
# ---------------------------------------------------------------------------


def _make_subject(volume: torch.Tensor) -> tio.Subject:
    affine = np.eye(4)
    return tio.Subject(input=tio.ScalarImage(tensor=volume, affine=affine))


def test_forward_preserves_shape_with_decimate() -> None:
    """``decimate=True`` returns the same Z-shape (decimate-then-NN-up)."""
    volume = torch.randn(1, 8, 8, 16)
    subj = _make_subject(volume)
    transform = SliceProfileTransform(
        profile="gauss",
        slice_thickness_mm=4.0,
        kernel_length=11,
        decimate=True,
        z_axis=3,
        keys=("input",),
    )
    out = transform.apply_transform(subj)
    assert out["input"].data.shape == volume.shape


def test_forward_preserves_shape_no_decimate() -> None:
    """``decimate=False`` returns the same shape (blur only)."""
    volume = torch.randn(1, 8, 8, 16)
    subj = _make_subject(volume)
    transform = SliceProfileTransform(
        profile="gauss",
        slice_thickness_mm=2.0,
        kernel_length=9,
        decimate=False,
        z_axis=3,
        keys=("input",),
    )
    out = transform.apply_transform(subj)
    assert out["input"].data.shape == volume.shape


def test_forward_skips_missing_keys() -> None:
    """Keys absent from the subject are silently skipped."""
    volume = torch.randn(1, 8, 8, 16)
    subj = _make_subject(volume)
    transform = SliceProfileTransform(
        profile="gauss",
        slice_thickness_mm=2.0,
        kernel_length=9,
        keys=("input", "ghost_key"),  # 'ghost_key' is not in subject
    )
    out = transform.apply_transform(subj)
    assert "input" in out
    assert "ghost_key" not in out


def test_slr_profile_runs_end_to_end() -> None:
    """SLR profile path is exercised end-to-end."""
    volume = torch.randn(1, 8, 8, 16)
    subj = _make_subject(volume)
    transform = SliceProfileTransform(
        profile="shinnar_le_roux",
        slice_thickness_mm=2.0,
        decimate=False,
        z_axis=3,
        keys=("input",),
    )
    out = transform.apply_transform(subj)
    assert out["input"].data.shape == volume.shape


#: The kwargs the one committed arm declares, pinned against the CORPUS rather
#: than the signature -- a default that drifts away from what arms write is the
#: failure this catches.
_ARM_KWARGS = {
    "profile": "shinnar_le_roux",
    "slice_thickness_mm": 4.0,
    "decimate": True,
    "z_axis": 3,
    "keys": ["input"],
}


class TestSliceProfileIsConfigDeclarable:
    """Registered, reachable, and reaching BOTH chains -- not just importable.

    ``SliceProfileTransform`` was never constructed anywhere in ``src/``. The one
    arm that wants it (``exp_slice_profile_sr.yaml``) names it by DOTTED PATH
    under ``data.augmentation.custom_transforms`` -- a key that is not a schema
    field, so ``extra="ignore"`` discards it at validation and the arm trains
    without the forward model it is named for.

    Registration fixes the half that is a code defect: there is now a canonical
    short name that resolves. The arm still has to move to
    ``data.processing.transforms`` for the chain to close; that is a config
    change, tracked separately.
    """

    def test_is_registered_under_the_short_name(self):
        from spectramr.data.transforms.registry import get_transform

        assert get_transform("slice_profile").cls is SliceProfileTransform

    def test_it_declares_that_it_adds_no_keys(self):
        """It rewrites ``input``/``target`` in place rather than adding a key,
        so an empty ``produces`` is the honest declaration -- not an oversight."""
        from spectramr.data.transforms.registry import get_transform

        assert get_transform("slice_profile").produces == ()

    def test_it_builds_with_the_kwargs_the_committed_arm_declares(self):
        """Pinned against the arm, not against the signature: a default that
        drifts away from what the corpus writes is the failure this catches."""
        from spectramr.data.transforms.registry import build_transform

        t = build_transform("slice_profile", **_ARM_KWARGS)
        assert isinstance(t, SliceProfileTransform)
        assert t.profile == "shinnar_le_roux"
        assert t.kernel_length == 33  # SLR is fixed-width; the arg is overridden

    @pytest.mark.parametrize("which", ["train", "val"])
    def test_a_declared_arm_reaches_the_built_chain(self, which):
        """The seam. Registry membership goes green the moment the decorator
        runs, while nothing constructs it -- which is the state this fixes."""
        from spectramr.config.schemas.data import DataProcessingConfigSchema
        from spectramr.data.builders.torchio_transform_builder import (
            TorchIOTransformBuilder,
            TorchIOTransformConfig,
        )
        from tests.utils.data_config_stub import DataConfigStub

        cfg = DataConfigStub(
            processing=DataProcessingConfigSchema(
                transforms=[{"name": "slice_profile", "kwargs": _ARM_KWARGS}]
            )
        )
        build = (
            TorchIOTransformBuilder.build_train_transforms
            if which == "train"
            else TorchIOTransformBuilder.build_val_transforms
        )
        compose = build(TorchIOTransformConfig.from_training_config(cfg))
        assert any(isinstance(t, SliceProfileTransform) for t in compose.transforms)

    def test_the_dotted_path_the_arm_uses_still_raises(self):
        """One canonical spelling. The dotted path never resolved and must not
        start now -- two spellings for one transform is the drift the registry
        exists to end. The error names the short name to migrate to."""
        from spectramr.data.transforms.registry import get_transform

        with pytest.raises(KeyError) as exc:
            get_transform("spectramr.data.transforms.slice_profile.SliceProfileTransform")
        assert "slice_profile" in str(exc.value)


class TestSliceProfileActsOnlyOnZ:
    """Correctness, probed with an impulse rather than noise.

    A shape/dtype assertion cannot tell a z-only slice profile from an isotropic
    blur, and white noise is invariant under too much to discriminate either. A
    single bright voxel makes the operator visible directly.
    """

    @staticmethod
    def _impulse_response(**kwargs):
        import torchio as tio

        from spectramr.data.transforms.registry import build_transform

        vol = torch.zeros(1, 8, 8, 16)
        vol[0, 4, 4, 8] = 1.0
        t = build_transform("slice_profile", **kwargs)
        return vol, t(tio.Subject(input=tio.ScalarImage(tensor=vol)))["input"].data

    def test_it_spreads_along_z_and_leaves_the_plane_a_delta(self):
        _, out = self._impulse_response(
            profile="gauss", slice_thickness_mm=4.0, decimate=False, keys=["input"]
        )
        assert out[0, 4, 4, 7] > 0, "no spread along z -- the profile did not fire"
        assert out[0, 3, 4, 8] == 0, "spread in-plane -- this is an isotropic blur"
        assert out[0, 4, 3, 8] == 0

    def test_the_z_response_is_symmetric(self):
        _, out = self._impulse_response(
            profile="gauss", slice_thickness_mm=4.0, decimate=False, keys=["input"]
        )
        assert out[0, 4, 4, 7] == pytest.approx(float(out[0, 4, 4, 9]), rel=1e-5)

    def test_energy_is_preserved(self):
        """A normalised profile redistributes signal; it must not scale it.
        A scale error here would read as a global intensity change downstream
        and be attributed to normalization."""
        vol, out = self._impulse_response(
            profile="gauss", slice_thickness_mm=4.0, decimate=False, keys=["input"]
        )
        assert float(out.sum()) == pytest.approx(float(vol.sum()), rel=1e-4)

    def test_a_key_the_subject_lacks_is_skipped_not_fatal(self):
        """``keys=['input']`` on a subject carrying only ``input`` must not
        require ``target`` to exist."""
        import torchio as tio

        from spectramr.data.transforms.registry import build_transform

        t = build_transform("slice_profile", keys=["input", "target"])
        s = tio.Subject(input=tio.ScalarImage(tensor=torch.rand(1, 4, 4, 8)))
        assert "input" in t(s)


# Captured from the pre-move slice_profile.py at commit b1428b57d: the output of
# SliceProfileTransform(profile, slice_thickness_mm=3.0) on rand(1, 1, 2, 12) (seed 0),
# one value per decimated slice (each repeats 3 times along z). Profiles differ by >= 0.06.
_GOLDEN_B1428B57D = {
    "sinc_gauss": [
        0.39484092593193054, 0.22085776925086975, 0.6116795539855957, 0.5172286033630371,
        0.06707239151000977, 0.5014984607696533, 0.3994806110858917, 0.6912357807159424,
    ],
    "gauss": [
        0.3436369299888611, 0.28191494941711426, 0.5914779901504517, 0.5127658843994141,
        0.08453066647052765, 0.483359694480896, 0.4552137851715088, 0.6454957127571106,
    ],
    "shinnar_le_roux": [
        0.17473796010017395, 0.2667175531387329, 0.43821290135383606, 0.419354647397995,
        0.07392837107181549, 0.3036385774612427, 0.443697065114975, 0.4545450210571289,
    ],
}  # fmt: skip


@pytest.mark.parametrize("profile", sorted(_GOLDEN_B1428B57D))
def test_output_matches_pre_move_golden(profile: str) -> None:
    vol = torch.rand(1, 1, 2, 12, generator=torch.Generator().manual_seed(0))
    subject = tio.Subject(input=tio.ScalarImage(tensor=vol))
    t = SliceProfileTransform(profile=profile, slice_thickness_mm=3.0, keys=("input",))
    out = t(subject)["input"].data
    expected = torch.tensor(_GOLDEN_B1428B57D[profile]).view(2, 4).repeat_interleave(3, dim=-1)
    # Bit-identical on the capturing machine; 1e-6 leaves room for another CPU's conv1d order.
    torch.testing.assert_close(out, expected.view(1, 1, 2, 12), rtol=0.0, atol=1e-6)
