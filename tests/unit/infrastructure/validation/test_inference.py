"""Unit tests for the recommendation + tag inference layer.

Mirrors test_data_model_compatibility.py's _SmallNamespace pattern so we
don't need a real Pydantic model for every case.
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.validation.inference import (
    collect_all_recommendations,
    derive_tags,
    recommend_diffusion_visualization,
    recommend_explicit_kspace_image_adapter,
    recommend_inverse_adapter,
    recommend_multi_contrast_alignment,
    recommend_strategy_paradigm_consistency,
)
from spectramr.infrastructure.validation.recommendations import (
    DIAGNOSTIC_CODES,
    Recommendation,
    RecommendationLevel,
    Tag,
)


@pytest.fixture(autouse=True)
def _drop_fake_registrations():
    """Remove this module's ``__test_*`` registrations after every test.

    ``_make_config`` registers into ``MODEL_REGISTRY``, which is PROCESS-GLOBAL
    and which ``populate_model_registry()`` cannot repair -- the
    ``_REGISTRY_POPULATED`` latch makes a second call a no-op, so a clear is
    permanent for the session. Without this teardown the fakes outlive the
    module and every later test file in the same process sees them.

    That is not hypothetical: ``object`` has no ``forward``, so the honour
    census in test_contrast_conditioned_sense_bridge.py raised
    ``AttributeError`` on the leaked entry and three of its tests were red
    whenever this file was collected first -- which ``pytest tests/unit/`` does,
    since ``infrastructure`` sorts before ``models``. They passed in isolation,
    so it read as flakiness. See #1961.
    """
    from spectramr.models.registry import MODEL_REGISTRY

    before = {n for n in MODEL_REGISTRY if n.startswith("__test_")}
    try:
        yield
    finally:
        for name in [n for n in MODEL_REGISTRY if n.startswith("__test_")]:
            if name not in before:
                MODEL_REGISTRY.pop(name, None)


class _NS:
    """Tiny attribute namespace stand-in for the audit's defensive getattr."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _make_config(
    *,
    patch_size=(64, 64, 1),
    dataset_type="nifti",
    multi_contrast_enabled=False,
    n_contrasts=None,
    acquisition_params=None,
    # Both real fields of `MultiContrastConfigSchema` that decide whether
    # `acquisition_params` is read at all. The stub omitted them, so every
    # R003 assertion was blind to the distinction between contrast-ID
    # conditioning (needs none) and Bloch physics conditioning (needs them) --
    # the same class of stub-vs-production gap the `sampling.patch_size` note
    # below records.
    embed_dim=None,
    bloch_grounded=False,
    model_type="cfg_unet",
    # Three-state since #1916: True declares support, False is a positive
    # claim that the model ignores the contrast id, and None means nobody
    # said. ``model_supports`` answers False for the latter two alike, so the
    # R003 polarity tests below are insensitive to that distinction; a test
    # that needs it must pass None explicitly rather than rely on the default.
    supports_contrast=None,
    model_kwargs=None,
    losses_output_domain="image",
    image_losses=(),
    kspace_losses=(),
    training_mode="reconstruction",
    strategy_class="spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
    save_validation_images=True,
    log_validation_images=True,
    adapters=None,
    physics_dc_enabled=False,
    physics_kspace_enabled=False,
    task="reconstruction",
):
    # Optionally register a fake model, THROUGH THE REAL PRODUCER.
    #
    # This used to hand-build the entry dict, and in doing so carried a
    # top-level ``supports_contrast_conditioning`` beside a nested
    # ``ModelCapabilities`` that had no such field -- reproducing inside the
    # fixture the exact two-owner split #1916 deleted from production. The
    # tests passed because fixture and reader agreed on the same wrong owner,
    # so they could not have caught the split and did not.
    #
    # A fixture that builds its own entry cannot notice when the producer's
    # shape changes. Going through ``register_model`` means it always has
    # whatever shape production writes -- including the ``role`` key (#1932),
    # which nothing here had to be told about.
    #
    # ``object`` is not an ``IDiscriminator``, so the default
    # ``role="generator"`` is correct and the registration guard stays quiet.
    # Re-registration across tests is fine: same class, non-empty caps.
    from spectramr.models.registry import register_model

    if model_type and model_type.startswith("__test_"):
        register_model(
            name=model_type,
            training_mode=training_mode,
            spatial_dims=(2, 3),
            input_domain="image",
            output_domain="image",
            supports_contrast_conditioning=supports_contrast,
        )(object)

    mc = _NS(
        enabled=multi_contrast_enabled,
        n_contrasts=n_contrasts,
        acquisition_params=acquisition_params,
        embed_dim=embed_dim,
        bloch_grounded=bloch_grounded,
    )
    data = _NS(
        dataset_type=dataset_type,
        # `data.sampling.patch_size` -- the decomposed shape the schema has. The
        # stub used to carry the flat leaf, which is why it kept passing while
        # the production read returned None on all 58 drained arms.
        sampling=_NS(patch_size=patch_size),
        multi_contrast=mc,
        coil_processing_mode="none",
        normalization_type="minmax",
        target_channels=1,
        in_channels=1,
        voxel_mm=None,
    )
    model = _NS(
        model_type=model_type,
        in_channels=1,
        out_channels=1,
        spatial_dims=len([d for d in patch_size if d > 1]),
        input_type="image",
        model_kwargs=model_kwargs or {},
    )
    # `losses.policy.output_domain`; the `*_losses` lists stayed top-level. The
    # production read is a direct attribute access, so the flat stub made
    # TestExplicitKspaceImageAdapter ERROR rather than assert (red before this).
    losses = _NS(
        policy=_NS(output_domain=losses_output_domain),
        image_losses=[_NS(name=n, weight=1.0, enabled=True) for n in image_losses],
        kspace_losses=[_NS(name=n, weight=1.0, enabled=True) for n in kspace_losses],
        complex_losses=[],
    )
    training = _NS(
        training_mode=training_mode,
        strategy_class=strategy_class,
        task=task,
    )
    # `logging.images.{save,log}_validation`. With the flat leaves both reads
    # returned False for every real config, so AUDIT-R005 fired unconditionally.
    logging_cfg = _NS(
        images=_NS(
            save_validation=save_validation_images,
            log_validation=log_validation_images,
        ),
    )
    physics = _NS(
        data_consistency=_NS(enabled=physics_dc_enabled),
        kspace=_NS(enable_kspace_recon=physics_kspace_enabled),
    )
    return _NS(
        data=data, model=model, losses=losses, training=training,
        logging=logging_cfg, physics=physics, adapters=adapters,
        metadata=None,
    )


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestDeriveTags:
    def test_no_special_attrs_yields_minimal_tags(self):
        cfg = _make_config()
        tags = derive_tags(cfg)
        names = {t.name for t in tags}
        assert "paradigm" in names
        assert "task" in names
        assert "spatial_dimensionality" in names
        assert "multi_contrast" not in names

    def test_multi_contrast_detected_from_data(self):
        cfg = _make_config(multi_contrast_enabled=True, n_contrasts=4)
        tags = {t.name: t for t in derive_tags(cfg)}
        assert tags["multi_contrast"].value is True
        assert "data.multi_contrast.enabled" in tags["multi_contrast"].derived_from

    def test_multi_contrast_detected_from_model_kwargs(self):
        cfg = _make_config(model_kwargs={"num_contrasts": 4, "use_contrast_guidance": True})
        tags = {t.name: t for t in derive_tags(cfg)}
        assert "multi_contrast" in tags

    def test_paradigm_inferred_from_strategy_class(self):
        cfg = _make_config(
            training_mode=None,
            strategy_class="spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
        )
        tags = {t.name: t for t in derive_tags(cfg)}
        assert tags["paradigm"].value == "diffusion"

    def test_spatial_dimensionality_2d_vs_3d(self):
        cfg2d = _make_config(patch_size=(64, 64, 1))
        cfg3d = _make_config(patch_size=(64, 64, 64))
        assert next(t for t in derive_tags(cfg2d) if t.name == "spatial_dimensionality").value == "2D"
        assert next(t for t in derive_tags(cfg3d) if t.name == "spatial_dimensionality").value == "3D"

    def test_physics_aware_tag(self):
        cfg = _make_config(physics_dc_enabled=True)
        tags = {t.name: t for t in derive_tags(cfg)}
        assert tags["physics_aware"].value is True


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class TestMultiContrastRecommendations:
    def test_r002_model_lacks_contrast_when_data_has_it(self):
        cfg = _make_config(
            model_type="__test_no_contrast",
            supports_contrast=False,
            multi_contrast_enabled=True,
            n_contrasts=4,
        )
        recs = recommend_multi_contrast_alignment(cfg)
        codes = {r.code for r in recs}
        assert "AUDIT-R002" in codes

    def test_r003_acquisition_params_missing(self):
        cfg = _make_config(
            model_type="__test_with_contrast",
            supports_contrast=True,
            multi_contrast_enabled=True,
            n_contrasts=4,
            acquisition_params=None,
        )
        recs = recommend_multi_contrast_alignment(cfg)
        codes = {r.code for r in recs}
        assert "AUDIT-R003" in codes

    def test_r010_model_wired_data_not(self):
        cfg = _make_config(
            multi_contrast_enabled=False,
            model_kwargs={"num_contrasts": 3, "use_contrast_guidance": True},
        )
        recs = recommend_multi_contrast_alignment(cfg)
        codes = {r.code for r in recs}
        assert "AUDIT-R010" in codes
        rec = next(r for r in recs if r.code == "AUDIT-R010")
        assert rec.suggested_yaml is not None
        assert "n_contrasts: 3" in rec.suggested_yaml

    def test_no_recommendation_when_aligned(self):
        cfg = _make_config(
            multi_contrast_enabled=True,
            n_contrasts=4,
            acquisition_params=[{"name": "T1w"}],
            model_kwargs={"num_contrasts": 4, "use_contrast_guidance": True},
            model_type="__test_with_contrast",
            supports_contrast=True,
        )
        codes = {r.code for r in recommend_multi_contrast_alignment(cfg)}
        assert "AUDIT-R002" not in codes
        assert "AUDIT-R003" not in codes
        assert "AUDIT-R010" not in codes

    def test_r010_exempts_mrixfields_native_contrast(self):
        # The mrixfields dataset emits a per-sample contrast_id natively (no
        # data.multi_contrast block needed), so a contrast-conditioned model
        # there is NOT dead weight — R010's premise is false. Do not misfire
        # (its suggested fix would wire a *different*, unused conditioning path).
        cfg = _make_config(
            dataset_type="mrixfields",
            multi_contrast_enabled=False,
            model_kwargs={"num_contrasts": 3, "use_contrast_conditioning": True},
        )
        codes = {r.code for r in recommend_multi_contrast_alignment(cfg)}
        assert "AUDIT-R010" not in codes

    # -- R003 applicability -------------------------------------------------
    #
    # `acquisition_params` feeds Bloch/PMPS *physics* conditioning. Contrast-ID
    # conditioning (Pattern C: `contrast_idx` -> nn.Embedding(n_contrasts,
    # embed_dim)) never reads TR/TE/TI/FA, so for an id-conditioned arm the old
    # blanket R003 asserted a fallback that does not happen AND shipped a
    # paste-ready block that would add a knob nothing reads (pitfall #15). The
    # genuine physics case already has its own higher-severity check (AUDIT-R021,
    # SUGGESTION), so the blanket form was redundant where it mattered and noise
    # everywhere else.
    #
    # Same reasoning, same file: `test_r010_exempts_mrixfields_native_contrast`
    # above declines to misfire because R010's suggested fix "would wire a
    # *different*, unused conditioning path". These are that test's polarity pair
    # for R003 -- and the second half exists because over-suppression is the
    # worse failure: a silenced hint on an arm that DOES read the params.

    def test_r003_reports_nothing_to_do_for_an_id_conditioned_arm(self):
        """Half 1 — the real `experiment_11_attention_none` shape."""
        cfg = _make_config(
            model_type="__test_with_contrast",
            supports_contrast=True,
            multi_contrast_enabled=True,
            n_contrasts=3,
            embed_dim=128,
            acquisition_params=None,
        )
        rec = next(
            r for r in recommend_multi_contrast_alignment(cfg) if r.code == "AUDIT-R003"
        )
        # Still emitted: the registry has no "consumes acquisition_params" flag,
        # so gating it off would have to guess across 590 models.
        assert "contrast INDEX" in rec.detail
        assert "No action needed" in rec.detail
        # The actively harmful half. A copy-paste block is an instruction; here
        # following it declares physics values nothing reads.
        assert rec.suggested_yaml is None, (
            "an id-conditioned arm was handed a paste-ready acquisition_params "
            "block; applying it would add an unread knob"
        )
        # The computed facts must be in the message, or "no action needed" is an
        # assertion the reader cannot check.
        assert "n_contrasts=3" in rec.detail
        assert "embed_dim=128" in rec.detail

    def test_r003_still_offers_the_yaml_when_bloch_grounded_reads_the_params(self):
        """Half 2 — over-suppression guard. Same config, `bloch_grounded` on."""
        cfg = _make_config(
            model_type="__test_with_contrast",
            supports_contrast=True,
            multi_contrast_enabled=True,
            n_contrasts=3,
            embed_dim=128,
            bloch_grounded=True,
            acquisition_params=None,
        )
        rec = next(
            r for r in recommend_multi_contrast_alignment(cfg) if r.code == "AUDIT-R003"
        )
        assert rec.suggested_yaml is not None, (
            "bloch_grounded=True consumes (TE, TR, TI, FA, B0) per contrast -- "
            "this arm must still be told to declare them"
        )
        assert "contrast INDEX" not in rec.detail
        assert "No action needed" not in rec.detail

    def test_r003_offers_the_yaml_when_the_model_is_not_contrast_registered(self):
        """Third polarity: id-conditioning is only inferable from the registry.

        A model with no `supports_contrast_conditioning` cannot be assumed to
        embed a contrast index, so the params may well be the intended path.
        """
        cfg = _make_config(
            model_type="__test_no_contrast",
            supports_contrast=False,
            multi_contrast_enabled=True,
            n_contrasts=3,
            acquisition_params=None,
        )
        rec = next(
            r for r in recommend_multi_contrast_alignment(cfg) if r.code == "AUDIT-R003"
        )
        assert rec.suggested_yaml is not None
        assert "No action needed" not in rec.detail

    def test_r003_labels_its_example_numbers_as_placeholders(self):
        """The snippet's TR/TE are shape-only and must not read as this protocol.

        Stamping invented physics into a YAML is worse than the missing block:
        it survives into provenance as a *declared* acquisition, indistinguishable
        from a measured one.
        """
        cfg = _make_config(
            model_type="__test_no_contrast",
            supports_contrast=False,
            multi_contrast_enabled=True,
            n_contrasts=3,
            acquisition_params=None,
        )
        rec = next(
            r for r in recommend_multi_contrast_alignment(cfg) if r.code == "AUDIT-R003"
        )
        assert "do not copy the example numbers" in rec.detail

    def test_r010_still_fires_for_non_native_dataset(self):
        # Guard against over-suppression: a non-mrixfields dataset with a
        # contrast-wired model and no multi_contrast block STILL gets R010.
        cfg = _make_config(
            dataset_type="nifti",
            multi_contrast_enabled=False,
            model_kwargs={"num_contrasts": 3, "use_contrast_guidance": True},
        )
        codes = {r.code for r in recommend_multi_contrast_alignment(cfg)}
        assert "AUDIT-R010" in codes


class TestDiffusionVisualization:
    def test_r005_fires_when_diffusion_without_image_save(self):
        cfg = _make_config(
            training_mode="diffusion",
            strategy_class="spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
            save_validation_images=False,
            log_validation_images=False,
        )
        recs = recommend_diffusion_visualization(cfg)
        assert any(r.code == "AUDIT-R005" for r in recs)

    def test_no_r005_when_image_save_enabled(self):
        cfg = _make_config(
            training_mode="diffusion",
            save_validation_images=True,
        )
        assert recommend_diffusion_visualization(cfg) == []

    def test_no_r005_for_non_diffusion(self):
        cfg = _make_config(
            training_mode="vae",
            save_validation_images=False,
            log_validation_images=False,
        )
        assert recommend_diffusion_visualization(cfg) == []


class TestExplicitKspaceImageAdapter:
    def test_r006_kspace_data_image_losses_no_adapter(self):
        cfg = _make_config(
            dataset_type="kspace",
            losses_output_domain="image",
            image_losses=("ssim",),
        )
        recs = recommend_explicit_kspace_image_adapter(cfg)
        assert any(r.code == "AUDIT-R006" for r in recs)


class TestStrategyParadigmConsistency:
    def test_r008_fires_on_mismatch(self):
        cfg = _make_config(
            training_mode="diffusion",
            strategy_class="spectramr.infrastructure.training.strategies.gan.GANTrainingStrategy",
        )
        recs = recommend_strategy_paradigm_consistency(cfg)
        assert any(r.code == "AUDIT-R008" for r in recs)

    def test_r008_silent_on_match(self):
        cfg = _make_config(
            training_mode="diffusion",
            strategy_class="spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
        )
        assert recommend_strategy_paradigm_consistency(cfg) == []


class TestInverseAdapter:
    def test_r001_fires_when_pre_model_invertible_without_post(self):
        # Need real adapters registered.
        import spectramr.data.adapters  # noqa: F401
        adapters = _NS(
            pre_model=[_NS(name="slice_3d_to_2d", enabled=True)],
            post_model=[],
            pre_loss_pred=[],
            pre_loss_target=[],
            pre_metric=[],
        )
        cfg = _make_config(adapters=adapters)
        recs = recommend_inverse_adapter(cfg)
        assert any(r.code == "AUDIT-R001" for r in recs)

    def test_r001_silent_when_inverse_present(self):
        import spectramr.data.adapters  # noqa: F401
        adapters = _NS(
            pre_model=[_NS(name="slice_3d_to_2d", enabled=True)],
            post_model=[_NS(name="gather_2d_to_3d", enabled=True)],
            pre_loss_pred=[], pre_loss_target=[], pre_metric=[],
        )
        cfg = _make_config(adapters=adapters)
        assert recommend_inverse_adapter(cfg) == []


class TestAggregator:
    def test_collect_all_recommendations_includes_each_generator(self):
        # Configuration that triggers two recommendations: R005 + R010
        cfg = _make_config(
            training_mode="diffusion",
            strategy_class="spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
            save_validation_images=False,
            log_validation_images=False,
            model_kwargs={"num_contrasts": 4, "use_contrast_guidance": True},
            multi_contrast_enabled=False,
        )
        recs = collect_all_recommendations(cfg)
        codes = {r.code for r in recs}
        assert "AUDIT-R005" in codes
        assert "AUDIT-R010" in codes

    def test_diagnostic_codes_have_titles(self):
        # Every code returned by a generator should be in the registry.
        cfg = _make_config(
            training_mode="diffusion",
            save_validation_images=False, log_validation_images=False,
        )
        recs = collect_all_recommendations(cfg)
        for r in recs:
            assert r.code in DIAGNOSTIC_CODES, f"{r.code} not in DIAGNOSTIC_CODES registry"


class TestRecommendationDataclass:
    def test_frozen(self):
        r = Recommendation(code="AUDIT-R001", title="x", detail="y")
        with pytest.raises(Exception):
            r.title = "z"  # type: ignore[misc]

    def test_levels_are_strings(self):
        for lvl in RecommendationLevel:
            assert isinstance(lvl.value, str)
