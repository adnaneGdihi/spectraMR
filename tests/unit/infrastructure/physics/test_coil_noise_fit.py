"""Sigma_n fitted from one scan's own null space.

The constant in ``m4raw_noise.py`` is scoped to 7 subjects from one study
series and says so: "Receiver gain may differ elsewhere in the corpus." R2R's
recorruption is exact only where ``Sigma_z == Sigma_n``, so every arm using it
inherits that scope. These tests measure against the real committed covariance
rather than a hand-written fixture, because the whole construction fails
quietly if that matrix is wrong by a scale factor.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.coil_noise_fit import (
    UnidentifiableCovarianceError,
    estimate_covariance_from_nullspace,
)
from spectramr.infrastructure.physics.coil_sensitivity import estimate_csm_espirit
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.m4raw_noise import single_repetition_covariance

_C, _H, _W = 4, 64, 64


@pytest.fixture(scope="module")
def phantom() -> dict[str, torch.Tensor]:
    """A rank-one multi-coil phantom and its ESPIRiT subspace."""
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _H), torch.linspace(-1, 1, _W), indexing="ij")
    magnitude = ((xx**2 + yy**2) < 0.6).float() * (1.0 + 0.3 * torch.cos(4 * xx))
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = torch.stack([torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres]).to(
        torch.complex64
    )
    sens = sens / sens.abs().pow(2).sum(0, keepdim=True).sqrt().clamp(min=1e-6)
    covariance = single_repetition_covariance().to(torch.complex64)
    clean = (sens * magnitude.to(torch.complex64)).unsqueeze(0) * (
        covariance.diagonal().real.mean().sqrt() * 8
    )
    subspace = estimate_csm_espirit(
        fft2c(clean[0]).unsqueeze(0), num_coils=_C, acs_size=24, return_subspace=True
    )
    return {
        "clean": clean,
        "covariance": covariance,
        "projector": subspace.null_projector,
        "support": subspace.leading_eigenvalue >= 0.95,
        "cholesky": torch.linalg.cholesky(covariance.to(torch.complex128)).to(torch.complex64),
    }


def _noise(cholesky: torch.Tensor) -> torch.Tensor:
    white = (torch.randn(_H * _W, _C) + 1j * torch.randn(_H * _W, _C)).to(torch.complex64)
    coloured = (white / 2**0.5) @ cholesky.transpose(-1, -2)
    return coloured.reshape(_H, _W, _C).permute(2, 0, 1).unsqueeze(0)


class TestPerScanCovariance:
    """Sigma_n estimated from the scan itself, not from a shipped constant.

    ``m4raw_noise.py`` scopes its covariance to 7 subjects from one study series
    and says receiver gain may differ elsewhere. R2R's recorruption is exact
    only where ``Sigma_z == Sigma_n``, so every arm using it inherits that
    scope. The null space is pure noise at every pixel, which is enough to fit
    the covariance per scan and retire the assumption.
    """

    @staticmethod
    def _projectors(directions: torch.Tensor) -> torch.Tensor:
        unit = directions / directions.norm(dim=-1, keepdim=True)
        eye = torch.eye(_C, dtype=torch.complex64)
        return eye - unit.unsqueeze(-1) @ unit.conj().unsqueeze(-2)

    @staticmethod
    def _noise_field(cholesky: torch.Tensor, n: int) -> torch.Tensor:
        white = (torch.randn(n, _C) + 1j * torch.randn(n, _C)).to(torch.complex64)
        return (white / 2**0.5) @ cholesky.transpose(-1, -2)

    def test_it_recovers_the_committed_covariance(self, phantom) -> None:
        torch.manual_seed(10)
        n = 3000
        noise = self._noise_field(phantom["cholesky"], n)
        images = noise.reshape(1, n, 1, _C).permute(0, 3, 1, 2)
        projectors = self._projectors(torch.randn(n, _C, dtype=torch.complex64)).reshape(
            1, n, 1, _C, _C
        )
        estimate = estimate_covariance_from_nullspace(images, projectors)
        truth = phantom["covariance"]
        assert (estimate - truth).abs().max() / truth.abs().max() < 0.12

    def test_it_is_hermitian_to_machine_precision(self, phantom) -> None:
        """Consumers assume it: ``expected_null_residual`` evaluates tr(Sigma P).

        The property holds by CONSTRUCTION -- the normal matrix is built from
        P (x) conj(P) with P Hermitian -- so the raw solve is already symmetric
        to ~1e-15 and the explicit symmetrisation is cosmetic. The tolerance is
        therefore tight on purpose: a loose one here would pass whatever the
        code did, which is what it did before this docstring was written.
        """
        torch.manual_seed(11)
        n = 2000
        images = self._noise_field(phantom["cholesky"], n).reshape(1, n, 1, _C).permute(0, 3, 1, 2)
        projectors = self._projectors(torch.randn(n, _C, dtype=torch.complex64)).reshape(
            1, n, 1, _C, _C
        )
        estimate = estimate_covariance_from_nullspace(images, projectors)
        assert (estimate - estimate.conj().transpose(-2, -1)).abs().max() < 1e-9

    def test_the_signal_is_rejected_across_the_working_range(self, phantom) -> None:
        """Amplitude must not move the estimate -- within the projector's accuracy.

        ESPIRiT's projector is ESTIMATED, so a little signal survives it and that
        residue scales with amplitude. Measured: flat at ~5.6% error while the
        signal is within an order of magnitude of the level the maps were
        estimated at, 10% at 20x, 41% at 50x. This pins the flat region, which is
        where real MRI SNR sits; ``test_the_rejection_degrades_far_above_it``
        pins that the limit is real rather than assumed away.
        """
        torch.manual_seed(12)
        noise = _noise(phantom["cholesky"])
        estimates = [
            estimate_covariance_from_nullspace(
                phantom["clean"] * k + noise, phantom["projector"], support=phantom["support"]
            )
            for k in (1, 10)
        ]
        drift = (estimates[0] - estimates[1]).abs().max() / estimates[0].abs().max()
        assert drift < 0.05, f"estimate moved {drift:.3f} over a 10x amplitude range"

    def test_the_rejection_degrades_far_above_it(self, phantom) -> None:
        """Anti-overclaim: the projector is not exact, and this says so.

        A test that only ever saw the flat region would let a future change
        present this as amplitude-independent, which it is not.
        """
        torch.manual_seed(12)
        noise = _noise(phantom["cholesky"])
        truth = phantom["covariance"]
        errors = [
            float(
                (
                    estimate_covariance_from_nullspace(
                        phantom["clean"] * k + noise,
                        phantom["projector"],
                        support=phantom["support"],
                    )
                    - truth
                )
                .abs()
                .max()
                / truth.abs().max()
            )
            for k in (1, 50)
        ]
        assert errors[1] > 3 * errors[0], errors

    def test_it_works_on_the_real_espirit_projector(self, phantom) -> None:
        torch.manual_seed(13)
        estimate = estimate_covariance_from_nullspace(
            phantom["clean"] + _noise(phantom["cholesky"]),
            phantom["projector"],
            support=phantom["support"],
        )
        truth = phantom["covariance"]
        assert (estimate - truth).abs().max() / truth.abs().max() < 0.15

    def test_no_coil_diversity_raises_rather_than_inventing_a_matrix(self, phantom) -> None:
        """One shared signal direction leaves that direction unconstrained.

        Returning a plausible matrix there is worse than refusing: the fit is
        blind to exactly the direction the array could not separate.
        """
        torch.manual_seed(14)
        n = 3000
        images = self._noise_field(phantom["cholesky"], n).reshape(1, n, 1, _C).permute(0, 3, 1, 2)
        shared = self._projectors(torch.randn(1, _C, dtype=torch.complex64))
        projectors = shared.reshape(1, 1, 1, _C, _C).expand(1, n, 1, _C, _C).contiguous()
        with pytest.raises(UnidentifiableCovarianceError, match="condition number"):
            estimate_covariance_from_nullspace(images, projectors)

    def test_too_few_pixels_raises(self, phantom) -> None:
        """C^2 unknowns need at least C^2 constraints."""
        torch.manual_seed(15)
        n = 4
        images = self._noise_field(phantom["cholesky"], n).reshape(1, n, 1, _C).permute(0, 3, 1, 2)
        projectors = self._projectors(torch.randn(n, _C, dtype=torch.complex64)).reshape(
            1, n, 1, _C, _C
        )
        with pytest.raises(UnidentifiableCovarianceError, match="cannot determine"):
            estimate_covariance_from_nullspace(images, projectors)

    def test_more_pixels_estimate_better(self, phantom) -> None:
        """Consistency. A biased estimator would not improve with N."""
        truth = phantom["covariance"]
        errors = []
        for n in (300, 12000):
            torch.manual_seed(16)
            images = (
                self._noise_field(phantom["cholesky"], n).reshape(1, n, 1, _C).permute(0, 3, 1, 2)
            )
            projectors = self._projectors(torch.randn(n, _C, dtype=torch.complex64)).reshape(
                1, n, 1, _C, _C
            )
            estimate = estimate_covariance_from_nullspace(images, projectors)
            errors.append(float((estimate - truth).abs().max() / truth.abs().max()))
        assert errors[1] < errors[0], errors


class TestTheScalarSigma:
    r"""``sigma_from_coil_nullspace``: one number, without the ``(C, C)`` tensor.

    Robust SSDU's ``noise_std`` is a scalar, and forming a per-pixel projector
    to get it would be ~0.8 s of eigendecomposition per slice inside a training
    step. For an idempotent projector ``E||P y||^2 = trace(Sigma P) ~ sigma^2
    (C-k)``, and ``||P y||^2 = ||y||^2 - |<s_hat, y>|^2``, so the projector is
    never formed and the cost is O(BCHW).
    """

    @staticmethod
    def _scene(gain: float, coils: int = 4, size: int = 128):
        """``(coil images, true maps, true magnitude sigma)`` at ``gain`` x M4Raw."""
        from spectramr.infrastructure.physics.m4raw_noise import single_repetition_covariance

        torch.manual_seed(0)
        cov = single_repetition_covariance().to(torch.complex64)
        chol = torch.linalg.cholesky(cov.to(torch.complex128)).to(torch.complex64)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
        )
        anatomy = (((xx**2 + yy**2) < 0.5).float() * (0.6 + 0.4 * torch.rand(size, size))).to(
            torch.complex64
        )
        centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)][:coils]
        maps = torch.stack([torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres]).to(
            torch.complex64
        )
        maps = maps * torch.exp(
            1j * torch.stack([k * xx for k in range(coils)]).to(torch.complex64)
        )
        maps = maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt().clamp(min=1e-6)
        white = (torch.randn(coils, size, size) + 1j * torch.randn(coils, size, size)).to(
            torch.complex64
        ) / (2**0.5)
        images = maps * anatomy * 30.0 + torch.einsum("ij,jhw->ihw", chol * gain, white)
        sigma = float(cov.diagonal().real.mean() ** 0.5) * gain
        return images.unsqueeze(0), maps.unsqueeze(0), sigma

    @pytest.mark.parametrize("gain", [0.5, 1.0, 2.0, 4.0])
    def test_it_recovers_a_known_sigma(self, gain: float) -> None:
        """Within 2% across an 8x span of noise, with exact maps."""
        from spectramr.infrastructure.physics.coil_noise_fit import sigma_from_coil_nullspace

        images, maps, truth = self._scene(gain)
        assert float(sigma_from_coil_nullspace(images, maps)) == pytest.approx(truth, rel=0.02)

    def test_it_is_insensitive_to_the_anatomy_brightness(self) -> None:
        """The whole premise: the null space carries noise and none of the signal."""
        from spectramr.infrastructure.physics.coil_noise_fit import sigma_from_coil_nullspace

        images, maps, truth = self._scene(1.0)
        brighter = images + maps.squeeze(0).unsqueeze(0) * 500.0
        assert float(sigma_from_coil_nullspace(brighter, maps)) == pytest.approx(truth, rel=0.05)

    def test_magnitude_maps_raise(self) -> None:
        """A real projector leaves the phase half of the null space in the residual."""
        from spectramr.infrastructure.physics.coil_noise_fit import sigma_from_coil_nullspace

        images, maps, _ = self._scene(1.0)
        with pytest.raises(ValueError, match="COMPLEX"):
            sigma_from_coil_nullspace(images, maps.abs())

    def test_a_single_coil_raises(self) -> None:
        """With one coil the null space is empty; there is nothing to measure."""
        from spectramr.infrastructure.physics.coil_noise_fit import sigma_from_coil_nullspace

        images, maps, _ = self._scene(1.0)
        with pytest.raises(ValueError, match="at least 2 coils"):
            sigma_from_coil_nullspace(images[:, :1], maps[:, :1])

    def test_a_shape_mismatch_raises(self) -> None:
        from spectramr.infrastructure.physics.coil_noise_fit import sigma_from_coil_nullspace

        images, maps, _ = self._scene(1.0)
        with pytest.raises(ValueError, match="per-pixel"):
            sigma_from_coil_nullspace(images[..., :64], maps)


class TestTheBatchWrapper:
    """``sigma_from_kspace_batch``: the shape the training step actually holds."""

    @staticmethod
    def _kspace_and_maps():
        from spectramr.infrastructure.physics.fft_ops import fft2c

        images, maps, truth = TestTheScalarSigma._scene(1.0)
        return fft2c(images), maps, truth

    def test_complex_and_interleaved_kspace_agree(self) -> None:
        """The loader serves real/imag interleaved; the strategy may hold either."""
        from spectramr.infrastructure.physics.coil_noise_fit import sigma_from_kspace_batch

        kspace, maps, _ = self._kspace_and_maps()
        interleaved = torch.zeros(1, 2 * kspace.shape[1], *kspace.shape[-2:])
        interleaved[:, 0::2] = kspace.real
        interleaved[:, 1::2] = kspace.imag
        assert float(sigma_from_kspace_batch(kspace, maps)) == pytest.approx(
            float(sigma_from_kspace_batch(interleaved, maps)), rel=1e-5
        )

    def test_it_measures_in_the_image_domain(self) -> None:
        """The rank-one coil model does not hold in k-space, so a k-space read
        would measure the anatomy rather than the noise."""
        from spectramr.infrastructure.physics.coil_noise_fit import (
            sigma_from_coil_nullspace,
            sigma_from_kspace_batch,
        )

        kspace, maps, _ = self._kspace_and_maps()
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        assert float(sigma_from_kspace_batch(kspace, maps)) == pytest.approx(
            float(sigma_from_coil_nullspace(ifft2c(kspace), maps)), rel=1e-6
        )

    def test_absent_maps_raise_rather_than_returning_a_constant(self) -> None:
        """A consumer asks for this BECAUSE it does not trust the constant."""
        from spectramr.infrastructure.physics.coil_noise_fit import sigma_from_kspace_batch

        kspace, _, _ = self._kspace_and_maps()
        with pytest.raises(ValueError, match="needs complex coil sensitivities"):
            sigma_from_kspace_batch(kspace, None)

    def test_combined_coils_raise(self) -> None:
        from spectramr.infrastructure.physics.coil_noise_fit import sigma_from_kspace_batch

        _, maps, _ = self._kspace_and_maps()
        with pytest.raises(ValueError, match="already combined"):
            sigma_from_kspace_batch(torch.randn(1, 3, 128, 128), maps)
