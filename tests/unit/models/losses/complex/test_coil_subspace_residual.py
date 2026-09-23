"""Energy off the coil manifold, and the ways this could be silently wrong.

The penalty is zero on every realisable image by construction, so the thing to
pin is not that it is small on good input -- that is trivially true of any term
multiplied by zero. It is that it is LARGE on input a coil array could not have
produced, and that it refuses to run on inputs where it would be quietly wrong:

* combined coils, where the coil vector no longer exists;
* MAGNITUDE sensitivity maps, which make ``P`` a real projector and drop the
  phase half of the null space without any error -- and one branch of the
  TorchIO subject builder really does store ``sensitivity`` as ``.abs()``;
* absent maps, where skipping would leave the term reading as "on" in the YAML
  while training without it.

The registration half matters as much as the arithmetic: a decorated loss whose
module is never imported is inert, and a domain that does not match the YAML
block it is declared in never gets built.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.complex.coil_subspace_residual import CoilSubspaceResidualLoss

_B, _C, _S = 2, 4, 32


def _maps() -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _S), torch.linspace(-1, 1, _S), indexing="ij")
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = torch.stack([torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres]).to(
        torch.complex64
    )
    # Phase that varies per coil: a magnitude-only map would lose exactly this.
    sens = sens * torch.exp(1j * torch.stack([k * xx for k in range(_C)]).to(torch.complex64))
    sens = sens / sens.abs().pow(2).sum(0, keepdim=True).sqrt().clamp(min=1e-6)
    return sens.unsqueeze(0).repeat(_B, 1, 1, 1)


def _anatomy() -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _S), torch.linspace(-1, 1, _S), indexing="ij")
    return ((xx**2 + yy**2) < 0.6).to(torch.complex64).unsqueeze(0).unsqueeze(0)


def _interleave(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(x.shape[0], 2 * x.shape[1], *x.shape[-2:])
    out[:, 0::2] = x.real
    out[:, 1::2] = x.imag
    return out


@pytest.fixture(scope="module")
def parts() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    maps = _maps()
    physical = maps * _anatomy()  # x = s * m, rank one in coil space
    return {"maps": maps, "physical": physical, "anatomy": _anatomy()}


class TestItMeasuresWhatItClaims:
    def test_a_physical_image_costs_nothing(self, parts) -> None:
        loss = CoilSubspaceResidualLoss(input_domain="kspace")
        value = loss(_interleave(fft2c(parts["physical"])), None, coil_sensitivities=parts["maps"])
        assert float(value) < 1e-6, value

    def test_energy_off_the_manifold_costs_a_lot(self, parts) -> None:
        """Seven orders of magnitude is the separation that makes it a barrier."""
        loss = CoilSubspaceResidualLoss(input_domain="kspace")
        clean = loss(_interleave(fft2c(parts["physical"])), None, coil_sensitivities=parts["maps"])
        broken = parts["physical"].clone()
        broken[:, 0] = broken[:, 0] + 0.3 * parts["anatomy"].squeeze(1)
        dirty = loss(_interleave(fft2c(broken)), None, coil_sensitivities=parts["maps"])
        assert float(dirty) > 1e4 * float(clean), (float(clean), float(dirty))

    def test_it_grows_with_the_invented_amplitude(self, parts) -> None:
        """Quadratic: it is a squared residual, not a hinge."""
        loss = CoilSubspaceResidualLoss(input_domain="kspace")
        values = []
        for scale in (0.1, 0.2):
            broken = parts["physical"].clone()
            broken[:, 0] = broken[:, 0] + scale * parts["anatomy"].squeeze(1)
            values.append(
                float(loss(_interleave(fft2c(broken)), None, coil_sensitivities=parts["maps"]))
            )
        assert 3.0 < values[1] / values[0] < 5.0, values

    def test_scaling_the_whole_image_does_not_change_the_direction(self, parts) -> None:
        """A rank-one image stays rank one however bright; the penalty stays ~0."""
        loss = CoilSubspaceResidualLoss(input_domain="kspace")
        bright = loss(
            _interleave(fft2c(parts["physical"] * 50)), None, coil_sensitivities=parts["maps"]
        )
        assert float(bright) < 1e-3, bright

    def test_the_image_domain_path_agrees_with_the_kspace_one(self, parts) -> None:
        broken = parts["physical"].clone()
        broken[:, 1] = broken[:, 1] + 0.25 * parts["anatomy"].squeeze(1)
        k = CoilSubspaceResidualLoss(input_domain="kspace")(
            _interleave(fft2c(broken)), None, coil_sensitivities=parts["maps"]
        )
        i = CoilSubspaceResidualLoss(input_domain="image")(
            _interleave(broken), None, coil_sensitivities=parts["maps"]
        )
        assert torch.allclose(k, i, rtol=1e-3, atol=1e-6), (float(k), float(i))

    def test_it_is_differentiable(self, parts) -> None:
        loss = CoilSubspaceResidualLoss(input_domain="kspace")
        pred = _interleave(fft2c(parts["physical"])).requires_grad_(True)
        loss(pred, None, coil_sensitivities=parts["maps"]).backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()


class TestItRefusesWhatWouldBeQuietlyWrong:
    def test_absent_maps_raise(self, parts) -> None:
        with pytest.raises(ValueError, match="requires coil_sensitivities"):
            CoilSubspaceResidualLoss()(_interleave(fft2c(parts["physical"])), None)

    def test_magnitude_maps_raise(self, parts) -> None:
        """The subject builder really does store `sensitivity` as `.abs()`.

        A real map makes P a real projector, which drops the phase half of the
        null space and under-penalises with no error anywhere.
        """
        with pytest.raises(ValueError, match="COMPLEX"):
            CoilSubspaceResidualLoss()(
                _interleave(fft2c(parts["physical"])),
                None,
                coil_sensitivities=parts["maps"].abs(),
            )

    def test_combined_coils_raise(self, parts) -> None:
        """An odd channel count means the coil vector is already destroyed."""
        combined = torch.randn(_B, 1, _S, _S)
        with pytest.raises(ValueError, match="interleaved"):
            CoilSubspaceResidualLoss()(combined, None, coil_sensitivities=parts["maps"])

    def test_a_shape_mismatch_raises(self, parts) -> None:
        with pytest.raises(ValueError, match="per-pixel"):
            CoilSubspaceResidualLoss()(
                _interleave(fft2c(parts["physical"]))[..., :16],
                None,
                coil_sensitivities=parts["maps"],
            )

    def test_all_zero_maps_raise(self, parts) -> None:
        """No pixel carries a rank-one constraint, so there is nothing to measure."""
        with pytest.raises(ValueError, match="every sensitivity map is zero"):
            CoilSubspaceResidualLoss()(
                _interleave(fft2c(parts["physical"])),
                None,
                coil_sensitivities=torch.zeros_like(parts["maps"]),
            )

    def test_an_unknown_input_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown input_domain"):
            CoilSubspaceResidualLoss(input_domain="latent")


class TestItIsReachableFromYaml:
    """A decorated loss nobody imports is inert; a mismatched domain never builds."""

    def test_the_module_is_imported_so_the_decorator_fires(self) -> None:
        """In a SUBPROCESS, because this file's own import would fire it.

        Asserting registration in-process is vacuous here: the module-level
        ``from ...coil_subspace_residual import`` at the top of this file
        registers the loss as a side effect, so the check passes even with the
        entry deleted from ``models/losses/__init__.py`` -- which is exactly the
        inert-registration state it exists to catch. Verified by planting that
        deletion: in-process stayed green, this does not.
        """
        import subprocess
        import sys

        probe = (
            "import spectramr.models.losses;"
            "from spectramr.models.losses.registry import LossRegistry;"
            "import sys;"
            "sys.exit(0 if 'coil_subspace_residual' in LossRegistry.list_available() else 1)"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, check=False)
        assert result.returncode == 0, (
            "coil_subspace_residual is not registered by importing "
            "spectramr.models.losses alone, so no YAML can select it.\n"
            + result.stderr.decode()[-500:]
        )

    def test_its_domain_matches_the_complex_losses_block(self) -> None:
        """`complex_losses` in YAML maps to domain `complex_image`, not `complex`."""
        from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS

        assert LOSS_LIST_DOMAINS["complex_losses"] == "complex_image"

    def test_the_constructor_signature_is_the_yaml_surface(self) -> None:
        """`_merged` raises on a kwarg the constructor does not take."""
        import inspect

        params = set(inspect.signature(CoilSubspaceResidualLoss.__init__).parameters)
        assert {"input_domain", "weight_by_support", "eps"} <= params


class TestItDeclaresItsOwnBridge:
    """The builder adds an iFFT bridge; two of them is silent, finite nonsense.

    ``complex_losses`` under ``output_domain: kspace`` is wrapped in an
    ``ifft_complex`` bridge by ``LossBuilder``. A term that ALSO bridges
    internally would be inverse-transformed twice: the outer bridge emits an
    image, the inner one re-reads it as k-space, re-pairs the channels as
    (real, imag) -- halving the coil count -- and iFFTs again. Nothing raises
    and the value is finite, so the run stays green while the term measures
    something else entirely (issue #467). The builder refuses that combination,
    but only for a loss that says so.
    """

    def test_the_kspace_domain_advertises_the_bridge(self) -> None:
        assert CoilSubspaceResidualLoss(input_domain="kspace").use_fourier_bridge is True

    def test_the_image_domain_does_not(self) -> None:
        """Declared under `complex_losses`, the builder's own bridge is the only one."""
        assert CoilSubspaceResidualLoss(input_domain="image").use_fourier_bridge is False

    def test_the_builder_rejects_the_double_bridge(self) -> None:
        """The guard reads the attribute, so an undeclared loss sails through."""
        from spectramr.infrastructure.training.builders import loss_builder as lb

        source = inspect.getsource(lb)
        assert 'getattr(loss_fn, "use_fourier_bridge", False)' in source, (
            "the double-bridge guard no longer reads the attribute this loss sets"
        )


class TestTheMapsReachItThroughTheProductionCall:
    """Registered and correct is the easy half; invoked WITH the maps is the change.

    ``UnifiedReconstructionLossComputer`` calls every declarative term through
    ``_call_safe_loss``, which filters kwargs by signature -- so the maps arrive
    only if the strategy puts them in the kwargs at all. They did not:
    ``coil_sensitivities`` reached ``_prepare_generator_inputs`` (the MODEL's
    forward) and stopped there.
    """

    def test_call_safe_loss_forwards_the_maps(self, parts) -> None:
        from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
            _call_safe_loss,
        )

        loss = CoilSubspaceResidualLoss(input_domain="image")
        value = _call_safe_loss(
            loss,
            _interleave(parts["physical"]),
            _interleave(parts["physical"]),
            coil_sensitivities=parts["maps"],
            pinn_loss=None,
            intermediate_outputs=None,
        )
        assert float(value) < 1e-6, value

    def test_without_them_the_production_call_still_raises(self, parts) -> None:
        """Loud, not silent: the term must not quietly evaluate to nothing."""
        from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
            _call_safe_loss,
        )

        with pytest.raises(ValueError, match="requires coil_sensitivities"):
            _call_safe_loss(
                CoilSubspaceResidualLoss(input_domain="image"),
                _interleave(parts["physical"]),
                _interleave(parts["physical"]),
                pinn_loss=None,
            )
