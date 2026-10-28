"""No physics option lets a bounded reverse loop accumulate energy.

Plain ``additive`` whitened the 8x and 32x rungs: it wrote each model estimate
into the state without a bound, and the model re-read and amplified those
writes. :class:`Feedback` is that model reduced to one line
(``2 * input + 5 max|y|``). The planted pair (non-negotiable 15):

* plain ``additive`` driven by it ends more than 10x over the magnitude ceiling;
* every bounded loop, under every physics option, ends under the ceiling the
  option declares, with the measured lines exactly pinned under hard DC.

``integrity_check='raise'`` runs throughout, so each executed step is also
verified (shape, finite, ceiling, pinned lines, and the ``3 sigma`` disc under
``fitted_decay``).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.diffusion.kspace_process import paired_magnitude
from spectramr.models.diffusion.reverse_hooks import prepare_reverse
from spectramr.models.diffusion.reverse_physics import HOOKED_REVERSE_MODES
from tests.unit.models.diffusion import _reverse_phantom as P

pytestmark = pytest.mark.unit

TOL = 1e-4
#: (decay_envelope, projection, dc_trust, dc_method)
OPTIONS = [
    ("off", "none", "measured", "hard"),
    ("per_sample", "none", "measured", "hard"),
    ("batch_pooled", "none", "measured", "hard"),
    ("off", "spirit", "measured", "hard"),
    ("off", "joint_tv", "measured", "hard"),
    ("per_sample", "none", "fitted_decay", "noise_adaptive"),
    ("batch_pooled", "joint_tv", "fitted_decay", "noise_adaptive"),
]


@pytest.fixture(scope="module")
def setting():
    proc = P.process(8.0)
    target = P.kspace()
    mask = P.mask(proc)
    return proc, target, target * mask, mask


def _feedback(proc, meas):
    return P.Feedback(proc, gain=2.0, bias=5.0 * float(meas.abs().max()))


def test_plain_additive_whitens_under_the_feedback_model(setting) -> None:
    proc, _, meas, mask = setting
    ceil = 1.5 * float(paired_magnitude(meas).max())
    out = P.sampler(_feedback(proc, meas), "additive").sample(meas, mask, start_timestep=P.HEAD)
    assert float(paired_magnitude(out).max()) > 10 * ceil


@pytest.mark.parametrize("mode", sorted(HOOKED_REVERSE_MODES))
@pytest.mark.parametrize(("decay", "projection", "trust", "dc"), OPTIONS)
def test_every_bounded_loop_stays_under_its_ceiling(
    setting, mode, decay, projection, trust, dc
) -> None:
    proc, _, meas, mask = setting
    sampler = P.sampler(
        _feedback(proc, meas),
        mode,
        dc_method=dc,
        decay_envelope=decay,
        projection=projection,
        dc_trust=trust,
        integrity_check="raise",
    )
    out = sampler.sample(meas, mask, start_timestep=P.HEAD)

    obs = sampler._observed_support(meas, mask, 1e-8)
    ctx = prepare_reverse(sampler, meas, obs, eps=1e-8)
    magnitude = paired_magnitude(out)
    unmeasured = obs.amax(dim=1, keepdim=True) == 0
    assert bool(torch.isfinite(out).all())
    assert bool((magnitude <= ctx.ceil * (1 + TOL) + TOL)[unmeasured.expand_as(magnitude)].all())
    assert float(magnitude.max()) <= 1.5 * float(paired_magnitude(meas).max()) * (1 + TOL)
    if dc == "hard":
        assert torch.equal(out * mask, meas), "measured lines must stay pinned"


@pytest.mark.parametrize("mode", sorted(HOOKED_REVERSE_MODES))
def test_the_decay_envelope_lowers_the_outer_rows(setting, mode) -> None:
    """With the envelope on, no unmeasured |ky| >= 1 row keeps the global ceiling."""
    proc, _, meas, mask = setting
    plain = P.sampler(_feedback(proc, meas), mode).sample(meas, mask, start_timestep=P.HEAD)
    decayed = P.sampler(_feedback(proc, meas), mode, decay_envelope="per_sample").sample(
        meas, mask, start_timestep=P.HEAD
    )
    unmeasured = (mask == 0).expand_as(paired_magnitude(meas))
    assert float(paired_magnitude(decayed)[unmeasured].max()) < float(
        paired_magnitude(plain)[unmeasured].max()
    )


def test_the_oracle_survives_every_bounded_loop_without_options(setting) -> None:
    """The bounds remove only energy the target does not have."""
    proc, target, meas, mask = setting
    for mode in sorted(HOOKED_REVERSE_MODES):
        out = P.sampler(P.Oracle(target, proc), mode, reverse_clip_ratio=100.0).sample(
            meas, mask, start_timestep=P.HEAD
        )
        assert torch.allclose(out, target, atol=1e-5), mode
