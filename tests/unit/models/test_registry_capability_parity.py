"""One class, one capability answer — a registry-wide ratchet (#1067).

``MODEL_REGISTRY`` maps a *name* to a class plus a declared capability set. Nothing
stops two names backed by the same class, or a trivial subclass of a registered class,
from declaring different capabilities — and when they do, the registry answers the same
question two ways depending on which spelling an arm happened to use.

That is not cosmetic. ``capabilities.spatial_dims`` and the domain fields are what
``check_workflow_spatial_rank``, the signal-domain checks and the audit spec card read.
A ``None`` there means UNDECLARED, so those checks *do not run*: an arm written against
the under-declared name silently loses pre-flight coverage that the identical arm
written against the base name receives.

The concrete failure this file was written for: ``latent_gaussian_diffusion`` is
``class LatentGaussianDiffusion(LatentDiffusionGenerator): pass`` — the same behaviour,
by construction — but its registration redeclared 1 of 6 capabilities. It reported
``spatial_dims=None`` while the base reported ``(2,)``. PR #1073 widened that gap
without noticing, by narrowing the base from ``(2, 3)`` to ``(2,)``.

These tests are deliberately written as a *ratchet*, not a cleanup: the known-divergent
entries are listed explicitly with a reason, so an existing divergence cannot be
mistaken for an accident, and any NEW one fails immediately.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY

#: Fields describing what the CLASS can do. ``training_mode`` is intentionally absent:
#: it is a registration fact (which training loop the name routes to), not a property of
#: the model, and several aliases exist precisely to register one class under a second
#: mode.
CAPABILITY_FIELDS = (
    "spatial_dims",
    "input_domain",
    "output_domain",
    "accepts_complex",
    "expects_real_imag_interleaved",
    "requires_paired_data",
    "output_field_units",
    "trajectory_parametrization",
)

#: Registered subclasses that legitimately declare LESS than their registered base, with
#: the reason. Everything here is a real divergence — it is excused, not fixed, because
#: forcing parity would mean asserting capabilities for models this change did not study,
#: and in this repo a WRONG declaration is a hard audit error while an absent one is only
#: advisory. Shrinking this dict is the point; growing it needs a reason in review.
KNOWN_UNDER_DECLARED: dict[str, str] = {
    # EMPTY — #1084 is fully drained. Kept as a mechanism, not as a habit: a future
    # subclass whose capabilities genuinely cannot be determined belongs here with a
    # reason, and the companion test below makes sure an excuse cannot outlive the
    # divergence it documents.
    #
    # gnn_coil_fusion was the last holdout, held back on the theory that a COIL-FUSION
    # model's input domain must differ from its graph_unet base. It does not — "coil" is
    # not a Domain at all ({image, kspace, complex_image, latent, pde_grid, mesh,
    # spectrum}); multi-coil-ness lives in the CHANNEL count. Its own arm
    # (experiment_78_gnn_coil.yaml) already declared input_domain: image over 32-coil
    # data, which settled it.
    #
    # The three MNO variants and mamba_unet_v2 were HERE until their ranks were measured
    # rather than guessed:
    #   * hs_/c_/s_mno_operator differ from cs_mno_operator only in the
    #     use_physical_arc / disable_spectral kwargs they force, and all three construct
    #     AND complete a forward at ranks 1 (raster_1d), 2 (hilbert_2d) and 3
    #     (hilbert_3d) — identical to the base, so they now declare its (1, 2, 3).
    #   * mamba_unet_v2 builds and forwards at rank 2 and fails at rank 3 in the UNet
    #     skeleton's conv2d, identical to mamba_unet, so it now declares (2,).
    # Kept as a note because "we could not tell" and "we checked" look the same in a
    # shrinking allowlist, and the difference is the whole point.
}

#: Same-class aliases whose declarations differ for a deliberate reason.
KNOWN_ALIAS_DIVERGENCE: dict[frozenset[str], str] = {
    frozenset({"graph_unet", "graph_unet_diffusion"}): (
        "same class registered under two training modes on purpose; only `mode` differs"
    ),
    frozenset(
        {
            "patch_gan",
            "patchgan_discriminator",
            "patch_latent_discriminator",
            "multiscale_latent_discriminator",
        }
    ): (
        "PatchGANDiscriminator scores two different input spaces, so `input_domain` is a "
        "property of the REGISTRATION and not of the class: the first two names score "
        "images, the two `latent` names score a post-VAE-encoder code. Collapsing them "
        'onto a permissive `("image", "kspace")` tuple would be accurate about tensor '
        "shapes and useless as a contract — `resolve_conversion` returns None as soon as "
        "any accepted side matches, so it would never convert. See the decorator block in "
        "models/discriminators/patchgan_discriminator.py, which is where the reason lives."
    ),
}


@pytest.fixture(scope="module", autouse=True)
def _populated_registry() -> None:
    """``MODEL_REGISTRY`` is EMPTY until explicitly populated, and a partial import
    order yields a partial registry — so this must run before any test reads it."""
    populate_model_registry()


def _capabilities(name: str) -> dict[str, object]:
    caps = MODEL_REGISTRY[name].get("capabilities")
    return {f: getattr(caps, f, None) for f in CAPABILITY_FIELDS}


def _class_of(name: str):
    entry = MODEL_REGISTRY[name]
    return entry["class"] if isinstance(entry, dict) else None


class TestSameClassAliasesAgree:
    """Two registry names backed by the SAME class object must describe it identically."""

    def test_no_unexpected_divergence_between_aliases_of_one_class(self) -> None:
        by_class = defaultdict(list)
        for name in MODEL_REGISTRY:
            cls = _class_of(name)
            if cls is not None:
                by_class[cls].append(name)

        offenders = []
        for cls, names in by_class.items():
            if len(names) < 2:
                continue
            if frozenset(names) in KNOWN_ALIAS_DIVERGENCE:
                continue
            distinct = {tuple(sorted(_capabilities(n).items())) for n in names}
            if len(distinct) > 1:
                differing = {
                    f for f in CAPABILITY_FIELDS if len({_capabilities(n)[f] for n in names}) > 1
                }
                offenders.append((cls.__name__, sorted(names), sorted(differing)))

        assert not offenders, (
            "These registry names share one class but describe it differently, so the "
            "registry answers the same question two ways depending on the spelling an "
            "arm used:\n" + "\n".join(f"  {c}: {ns} differ on {fs}" for c, ns, fs in offenders)
        )


class TestSubclassesDoNotSilentlyDropCapabilities:
    """A registered subclass of a registered class must not leave a field UNDECLARED
    that its base declares — ``None`` disables the audit checks that read it."""

    @staticmethod
    def _under_declared() -> dict[str, dict]:
        name_of_class = {}
        for name in MODEL_REGISTRY:
            cls = _class_of(name)
            if cls is not None:
                name_of_class.setdefault(cls, name)

        found: dict[str, dict] = {}
        for name in MODEL_REGISTRY:
            cls = _class_of(name)
            if cls is None:
                continue
            for base in cls.__mro__[1:]:
                if base in name_of_class and name_of_class[base] != name:
                    base_name = name_of_class[base]
                    child, parent = _capabilities(name), _capabilities(base_name)
                    lost = [
                        f for f in CAPABILITY_FIELDS if child[f] is None and parent[f] is not None
                    ]
                    if lost:
                        found[name] = {"base": base_name, "lost": lost}
                    break
        return found

    def test_the_1067_alias_now_mirrors_its_base(self) -> None:
        """``latent_gaussian_diffusion`` is a ``pass`` subclass — same behaviour, so it
        must give the same answers. This is the regression the file exists for."""
        alias = _capabilities("latent_gaussian_diffusion")
        base = _capabilities("latent_gan_generator")
        assert alias == base, (
            "latent_gaussian_diffusion is `class LatentGaussianDiffusion"
            "(LatentDiffusionGenerator): pass`, so any capability difference is a bug."
        )

    def test_the_1067_alias_reports_the_narrowed_2d_only_spatial_dims(self) -> None:
        """Pins the specific value PR #1073 narrowed on the base and not on the alias:
        AutoencoderKL's submodules are hardcoded ``nn.Conv2d`` (#1063)."""
        assert _capabilities("latent_gaussian_diffusion")["spatial_dims"] == (2,)

    def test_no_new_subclass_silently_drops_a_capability(self) -> None:
        """The ratchet. Any NEW under-declaring subclass fails here; the existing ones
        are excused by name in ``KNOWN_UNDER_DECLARED`` with a reason."""
        unexpected = {
            n: info for n, info in self._under_declared().items() if n not in KNOWN_UNDER_DECLARED
        }
        assert not unexpected, (
            "These registered subclasses leave a capability UNDECLARED that their "
            "registered base declares. `None` means the audit checks reading it do not "
            "run, so an arm using this name loses pre-flight coverage the base name "
            "gets. Mirror the base's declaration, or add it to KNOWN_UNDER_DECLARED "
            "with a reason:\n"
            + "\n".join(
                f"  {n} (base {i['base']}) loses {i['lost']}" for n, i in sorted(unexpected.items())
            )
        )

    def test_the_allowlist_does_not_outlive_its_entries(self) -> None:
        """A stale excuse is worse than none: it documents a divergence that no longer
        exists and hides the next one behind an out-of-date reason."""
        stale = sorted(set(KNOWN_UNDER_DECLARED) - set(self._under_declared()))
        assert not stale, (
            f"KNOWN_UNDER_DECLARED excuses {stale}, which no longer under-declare. "
            "Remove the entries — the ratchet should tighten."
        )
