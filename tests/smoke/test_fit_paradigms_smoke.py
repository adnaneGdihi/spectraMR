"""End-to-end smoke verification for ``spectramr.pipelines.fit.fit`` paradigms.

This is the **non-facade verifier**: it runs each paradigm's ``fit(...)`` for a
couple of real iterations on a tiny synthetic ``(input, target)`` loader with a
paradigm-appropriate model, and asserts the run completes AND the paradigm's
*distinctive* loss key actually appears in the result (a present-but-absent
distinctive loss is the silent-collapse facade the codebase polices — pitfalls
#10/#16). Adding a tailored ``fit()`` default for a new paradigm should come with
an entry here.

.. warning::

   These run a real training loop (DI container + model + loop) and are **heavy**
   — run them on the cluster, NOT on a dev box (a single fit() loop can OOM-kill
   an interactive machine). They are marked ``smoke`` so they stay out of the
   fast unit suite; invoke explicitly with ``pytest tests/smoke/test_fit_paradigms_smoke.py``.

The harness is paradigm-agnostic: each case supplies ``(paradigm, model_factory,
distinctive_loss_substr, extra_fit_kwargs)``. Reconstruction is the baseline that
proves the harness itself; the rest extend it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from spectramr.pipelines.fit import _PARADIGM_DEFAULTS  # noqa: E402

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


class _SynthPairs(Dataset):
    """Tiny synthetic dataset yielding the canonical ``(input, target)`` mapping."""

    def __init__(self, n: int = 4, c: int = 1, h: int = 16, w: int = 16) -> None:
        self.x = torch.randn(n, c, h, w)
        self.y = torch.randn(n, c, h, w)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {"input": self.x[i], "target": self.y[i]}


def _loader(batch_size: int = 2, size: int = 16) -> DataLoader:
    return DataLoader(_SynthPairs(h=size, w=size), batch_size=batch_size)


def _run_fit_smoke(
    paradigm: str, model, *, distinctive=None, input_size: int = 16, **fit_kwargs
) -> dict:
    """Run a tiny real fit() and return the result dict (heavy — cluster only)."""
    from spectramr.pipelines.fit import fit

    result = fit(
        model,
        _loader(size=input_size),
        paradigm=paradigm,
        device="cpu",
        max_iterations=2,
        # This harness inspects the result dict itself so a failure reports the
        # paradigm name and the pipeline's own error text. `fit` defaults to
        # raising (a script has no exit code to read); here the assertion below
        # is the better diagnostic.
        raise_on_failure=False,
        **fit_kwargs,
    )
    assert isinstance(result, dict), f"{paradigm}: fit returned {type(result)}"
    assert result.get("success") is True, f"{paradigm}: fit did not succeed — {result.get('error')}"
    if distinctive is not None:
        blob = (
            " ".join(str(k) for k in result)
            + " "
            + str(result.get("best_metrics") or {})
            + " "
            + str(result.get("final_loss") or "")
        )
        assert distinctive in blob, (
            f"{paradigm}: distinctive loss/metric {distinctive!r} not found in "
            f"result — possible facade (strategy ran but its mechanism did not "
            f"fire). result keys={sorted(result)}"
        )
    return result


# ---------------------------------------------------------------------------
# Paradigm cases. Reconstruction is the baseline (a plain conv suffices); add a
# case per tailored paradigm as its compatible model factory is confirmed.
# ---------------------------------------------------------------------------


class _TimestepConv(nn.Module):
    """A tiny model that accepts a diffusion timestep (positional or kwarg)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


class _VIBConv(nn.Module):
    """The minimal encode/reparameterize/decode surface the VIB objective needs.

    ``RecoverabilityVIBStrategy`` computes its information-bottleneck Lagrangian
    from ``model.encode(x) -> (mu, logvar)``, ``model.reparameterize(...)`` and
    ``model.decode(z, size)``. A plain ``nn.Conv2d`` has none of them, which is
    why this case failed with ``'Conv2d' object has no attribute 'encode'`` --
    a fixture that never matched the paradigm, not a framework defect.

    Mirrors ``_TimestepConv``: the smallest model that satisfies the paradigm's
    contract, so the case tests the STRATEGY rather than the stub.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Conv2d(1, 2, 3, padding=1)  # -> (mu, logvar) channels
        self.dec = nn.Conv2d(1, 1, 3, padding=1)

    def encode(self, x):
        h = self.enc(x)
        return h[:, :1], h[:, 1:]

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z, size=None):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar), x.shape[-2:])


@pytest.mark.parametrize("paradigm", ["ssl", "mae", "physics_equivariant_ssl"])
def test_ssl_family_smoke(paradigm):
    """SSL pretraining passes extra kwargs into ``forward``; a bare conv cannot.

    ``_TimestepConv``'s ``*args, **kwargs`` is what makes it usable here -- the
    shared ``nn.Conv2d`` fixture raises ``Conv2d.forward() got an unexpected
    keyword argument``, which is a fixture mismatch, not a strategy defect.
    """
    _run_fit_smoke(paradigm, _TimestepConv())


@pytest.mark.parametrize("paradigm", ["vae", "vqvae"])
def test_vae_family_smoke(paradigm):
    """VAE/VQ-VAE need an encoder-bearing model, like recoverability_vib."""
    _run_fit_smoke(paradigm, _VIBConv())


def test_recoverability_vib_smoke():
    """VIB needs an encoder-bearing model; the shared conv fixture cannot serve it."""
    _run_fit_smoke("recoverability_vib", _VIBConv())


def test_reconstruction_smoke():
    """Baseline: fit(reconstruction) runs a real loop on synthetic pairs."""
    _run_fit_smoke("reconstruction", nn.Conv2d(1, 1, 3, padding=1))


@pytest.mark.parametrize(
    "paradigm",
    [
        "masked",
        "cartoon_texture_safe",
        "generative",
    ],
)
def test_recon_compatible_paradigms_smoke(paradigm):
    """Paradigms verified (static) to run with a standard image model on
    ``(input, target)`` pairs. The cluster run is the live non-facade check."""
    _run_fit_smoke(paradigm, nn.Conv2d(1, 1, 3, padding=1))


@pytest.mark.parametrize(
    "paradigm",
    [
        "diffusion",
        "edm",
        "x_diffusion",
        "cross_modal_diffusion",
        "flow_matching_pfode",
        "flow_matching",
        "rectified_flow",
        "fisher_rao_flow",
        "levy_diffusion",
        "resetting_diffusion",
        "tissue_diffusion_pretrain",
        # Advertised in ``_PARADIGM_ALIASES`` (-> diffusion) but covered by
        # nothing, and broken: its ``isinstance(batch, dict) else batch`` handed
        # the whole TrainingBatch on as the target tensor. A paradigm ``fit``
        # offers with a tailored default belongs in this list.
        "stochastic_interpolants",
    ],
)
def test_diffusion_family_smoke(paradigm):
    """Diffusion-family paradigms — the model must accept a timestep. The cluster
    run verifies the distinctive (velocity / score / EDM) loss actually fires."""
    _run_fit_smoke(paradigm, _TimestepConv())


def test_gan_smoke_adversarial_term_actually_fires():
    """fit(gan) must train ADVERSARIALLY, not collapse to a vanilla L1 denoiser.

    Wires the previously-dead ``distinctive=`` anti-facade hook for the most
    facade-prone paradigm: the B2 regression (``enable_adversarial`` missing from
    the fit gan default) built a discriminator but trained only L1, and this
    case is what would have caught it pre-merge — it asserts an adversarial term
    appears in the result. The substring is validated on the cluster; if the GAN
    strategy names its adversarial metric differently, adjust ``distinctive`` to
    that key (the unit guard is ``test_fit.py``'s enable_adversarial assertion).
    """
    _run_fit_smoke(
        "gan",
        nn.Conv2d(1, 1, 3, padding=1),
        discriminator=nn.Conv2d(1, 1, 3, padding=1),
        # ``d_total_loss`` is the strongest adversarial evidence VISIBLE HERE:
        # the run report aggregates totals only, so no per-component key (the
        # generator's ``g_adv_loss``) ever reaches ``best_metrics``. The previous
        # ``"adv"`` was validated against nothing, because this case could not
        # get far enough to report any metric at all.
        #
        # The component-level anti-facade assertion -- that the GENERATOR
        # actually receives an adversarial term, which is the L1 collapse this
        # case names -- lives in
        # ``tests/unit/models/losses/computers/test_unified_gan_weighting.py``,
        # where it can inspect the loss output directly instead of grepping an
        # aggregated blob.
        distinctive="d_total_loss",
        # 16x16 is too small for the VGG perceptual term inside
        # ``CompositeGANLoss`` ("Calculated output size: (512x0x0)"), which
        # only becomes reachable once the adversarial path actually fires.
        input_size=64,
        # LOAD-BEARING. ``adversarial`` is in ``LEGACY_WARMUP_LOSSES`` and
        # ``warmup_iterations`` defaults to 1000, so over this run's 2 iterations
        # the adversarial weight resolves to 0.0 and the generator trains on
        # reconstruction alone -- the exact L1 collapse this case exists to
        # catch. Warm-up is correct behaviour for a real run; a 2-iteration smoke
        # test that wants to observe the mechanism has to step outside it, and
        # say so, rather than assert against a regime where it cannot fire.
        #
        # Merged onto the paradigm default rather than replacing it:
        # ``_resolve_fit_config`` does ``base.setdefault("losses", ...)``, so a
        # partial ``losses`` dict discards the whole default block -- including
        # the ``losses.gan`` the GAN strategy requires. Here that raises; for a
        # paradigm whose distinctive term is merely gated, it would collapse
        # silently.
        config={
            "losses": {
                **_PARADIGM_DEFAULTS["gan"]["losses"],
                "reconstruction": {"warmup_iterations": 0},
            }
        },
    )


def test_every_advertised_paradigm_has_a_smoke_case():
    """``fit`` must not advertise a paradigm nothing exercises.

    ``_PARADIGM_DEFAULTS`` + ``_PARADIGM_ALIASES`` are a PROMISE -- each key ships
    a tailored default so ``fit(paradigm=...)`` "just works". This matrix covered
    13 of 24, and four of the six batch-accessor leaks fixed alongside this test
    were in paradigms with no case at all. ``stochastic_interpolants`` raised on
    every run while being advertised; ``ssl`` and ``mae`` failed validation
    outright.

    A source-level check, because the promise is about the TABLE, not about any
    one run: adding a key to ``_PARADIGM_DEFAULTS`` without adding a case here
    should fail immediately, at authoring time.
    """
    import inspect
    import re

    from spectramr.pipelines.fit import _PARADIGM_ALIASES, _PARADIGM_DEFAULTS

    advertised = set(_PARADIGM_DEFAULTS) | set(_PARADIGM_ALIASES)
    source = inspect.getsource(inspect.getmodule(test_reconstruction_smoke))
    covered = {name for name in re.findall(r'"([a-z_0-9]+)"', source) if name in advertised}

    assert not advertised - covered, (
        "paradigm(s) advertised by fit() with no smoke case — each is a tailored "
        "default nothing has ever executed:\n  " + "\n  ".join(sorted(advertised - covered))
    )
