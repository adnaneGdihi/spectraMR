import pytest
from pydantic import ValidationError

from spectramr.config.schemas.base import (
    BiophysicalFlowObjectiveConfig,
    DiffusionObjectiveConfig,
    DomainAdaptationObjectiveConfig,
    EncoderDecoderDefinitionSchema,
    ExperimentMetadataSchema,
    FSDPConfigSchema,
    GANObjectiveConfig,
    LatentObjectiveConfig,
    ModuleDefinitionSchema,
    MultiContrastContrastiveConfigSchema,
    ParallelismConfigSchema,
    PEFTConfigSchema,
    R1RegularizationConfig,
    ReconstructionObjectiveConfig,
    ServicesConfigSchema,
    SSLConfigSchema,
    SSLObjectiveConfig,
    TrainingTaskConfigSchema,
)


class TestBaseSchemas:
    def test_module_definition_schema(self):
        schema = ModuleDefinitionSchema(name="test", params={"a": 1})
        assert schema.name == "test"
        assert schema.params == {"a": 1}
        assert schema.freeze is False

    def test_encoder_decoder_definition_schema(self):
        encoder = ModuleDefinitionSchema(name="enc")
        decoder = ModuleDefinitionSchema(name="dec")
        schema = EncoderDecoderDefinitionSchema(encoder=encoder, decoder=decoder)
        assert schema.encoder.name == "enc"
        assert schema.decoder.name == "dec"

    def test_experiment_metadata_schema(self):
        schema = ExperimentMetadataSchema(name="exp", tags=["tag1", "tag2"])
        assert schema.name == "exp"
        assert schema.tags == {"tag1": "tag1", "tag2": "tag2"}

    def test_experiment_metadata_scientific_fields(self):
        """First-class scientific-coherence metadata (2026-06 validation campaign)."""
        # default to None so pre-existing configs are unaffected
        bare = ExperimentMetadataSchema(name="exp")
        assert bare.hypothesis is None
        assert bare.baseline is None
        assert bare.primary_metric is None
        # and are accepted as first-class keys (no longer need to hide in tags)
        schema = ExperimentMetadataSchema(
            name="exp",
            hypothesis="method X beats baseline Y on val_psnr at 4x",
            baseline="experiment_baseline_unet_4x",
            primary_metric="val_psnr",
        )
        assert schema.hypothesis == "method X beats baseline Y on val_psnr at 4x"
        assert schema.baseline == "experiment_baseline_unet_4x"
        assert schema.primary_metric == "val_psnr"

    # --- metadata.status: closed vocabulary, one spelling (cohort review 2026-09-02, T0.4)

    def test_experiment_status_defaults_to_none_and_accepts_the_vocabulary(self):
        from spectramr.config.schemas.base import EXPERIMENT_STATUSES

        assert ExperimentMetadataSchema(name="exp").status is None
        for status in EXPERIMENT_STATUSES:
            assert ExperimentMetadataSchema(name="exp", status=status).status == status

    def test_experiment_status_free_text_is_refused(self):
        """The planted violation: the corpus carried 21 free-text tokens nothing read."""
        with pytest.raises(ValidationError):
            ExperimentMetadataSchema(name="exp", status="testable_on_real_b0")

    def test_experiment_status_reason_carries_the_free_text(self):
        schema = ExperimentMetadataSchema(
            name="exp", status="needs_data", status_reason="testable_on_real_b0"
        )
        assert schema.status_reason == "testable_on_real_b0"

    def test_tags_status_spelling_is_retired(self):
        """111 arms wrote the status where nothing typed could read it."""
        with pytest.raises(ValidationError, match="tags.status is retired"):
            ExperimentMetadataSchema(name="exp", tags={"status": "needs_implementation"})

    def test_launch_refused_statuses_are_a_subset_of_the_vocabulary(self):
        from spectramr.config.schemas.base import EXPERIMENT_STATUSES, LAUNCH_REFUSED_STATUSES

        assert LAUNCH_REFUSED_STATUSES == {"needs_implementation", "inert", "blocked"}
        assert LAUNCH_REFUSED_STATUSES <= set(EXPERIMENT_STATUSES)

    def test_free_form_prose_keys_are_kept(self):
        """``extra="allow"``: ~60 prose keys live in the corpus (note, group, ...)."""
        schema = ExperimentMetadataSchema(name="exp", note="free text", group="g1")
        assert schema.model_dump()["note"] == "free text"

    def test_scalar_version_and_date_are_stringified(self):
        import datetime

        schema = ExperimentMetadataSchema(
            name="exp", version=6.0, created_at=datetime.date(2026, 4, 10)
        )
        assert schema.version == "6.0" and schema.created_at == "2026-04-10"

    def test_r1_regularization_config(self):
        schema = R1RegularizationConfig(weight=5.0)
        assert schema.weight == 5.0
        with pytest.raises(ValidationError):
            R1RegularizationConfig(weight=-1.0)

    def test_gan_objective_config(self):
        # Lambda fields have been moved to losses.gan (SSOT consolidation)
        # v6.0: objectives.gan is NO LONGER supported and raises ConfigurationError
        from spectramr.domain.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            GANObjectiveConfig()

    def test_reconstruction_objective_config(self):
        # Lambda fields have been moved to losses.reconstruction (SSOT consolidation)
        # v6.0: objectives.reconstruction is NO LONGER supported and raises ConfigurationError
        from spectramr.domain.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            ReconstructionObjectiveConfig()

    def test_diffusion_objective_config(self):
        # DiffusionObjectiveConfig no longer accepts timesteps (DEPRECATED)
        # Timesteps is now SSOT in training.diffusion
        # Test that objectives.diffusion accepts the fields that remain
        # NOTE: DiffusionObjectiveConfig does NOT raise ConfigurationError yet in base.py
        schema = DiffusionObjectiveConfig(cond_drop_prob=0.1)
        assert schema.cond_drop_prob == 0.1
        with pytest.raises(ValidationError):
            DiffusionObjectiveConfig(cond_drop_prob=1.5)

    def test_latent_objective_config(self):
        # LatentObjectiveConfig.latent_dim is DEPRECATED (SSOT: training.latent_dim)
        # v6.0: objectives.latent is NO LONGER supported and raises ConfigurationError
        from spectramr.domain.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            LatentObjectiveConfig(n_embeddings=512)

    def test_domain_adaptation_objective_config(self):
        schema = DomainAdaptationObjectiveConfig(num_domains=3)
        assert schema.num_domains == 3
        with pytest.raises(ValidationError):
            DomainAdaptationObjectiveConfig(num_domains=1)

    def test_biophysical_flow_objective_config(self):
        schema = BiophysicalFlowObjectiveConfig(lambda_dc=2.0)
        assert schema.lambda_dc == 2.0
        with pytest.raises(ValidationError):
            BiophysicalFlowObjectiveConfig(lambda_dc=-1.0)

    def test_ssl_objective_config(self):
        schema = SSLObjectiveConfig(student_temp=0.2)
        assert schema.student_temp == 0.2
        with pytest.raises(ValidationError):
            SSLObjectiveConfig(student_temp=0.0)

    def test_ssl_config_schema(self):
        schema = SSLConfigSchema(method="simclr")
        assert schema.method == "simclr"

    def test_multi_contrast_contrastive_config_defaults(self):
        cfg = MultiContrastContrastiveConfigSchema()
        assert cfg.temperature > 0
        # Default keeps symmetrize off so it does not conflict with the
        # default-on momentum encoder (the strategy raises on the combo).
        assert cfg.symmetrize is False
        assert cfg.use_momentum is True
        assert 0 <= cfg.momentum <= 1
        assert cfg.warmup_steps >= 0

    def test_multi_contrast_contrastive_validation(self):
        with pytest.raises(ValidationError):
            MultiContrastContrastiveConfigSchema(unknown_field=1)
        with pytest.raises(ValidationError):
            MultiContrastContrastiveConfigSchema(temperature=0.0)
        with pytest.raises(ValidationError):
            MultiContrastContrastiveConfigSchema(momentum=1.5)

    def test_ssl_config_mounts_multi_contrast_block(self):
        ssl = SSLConfigSchema(multi_contrast_contrastive={"temperature": 0.07, "symmetrize": False})
        assert ssl.multi_contrast_contrastive is not None
        assert ssl.multi_contrast_contrastive.temperature == 0.07
        # Defaults to None when omitted.
        assert SSLConfigSchema().multi_contrast_contrastive is None

    def test_the_artifact_schema_is_gone(self):
        """`artifacts` was removed (#698): the key is refused at load, so no schema models it."""
        from spectramr.config.schemas import base

        assert not hasattr(base, "ArtifactConfig")

    def test_services_config_schema(self):
        schema = ServicesConfigSchema(enable_logging=False)
        assert schema.enable_logging is False

    def test_training_task_config_schema(self):
        schema = TrainingTaskConfigSchema(sr_factor=4)
        assert schema.sr_factor == 4
        with pytest.raises(ValidationError):
            TrainingTaskConfigSchema(sr_factor=0)

    def test_parallelism_config_schema(self):
        schema = ParallelismConfigSchema(num_devices=2)
        assert schema.num_devices == 2
        with pytest.raises(ValidationError):
            ParallelismConfigSchema(num_devices=0)

    def test_fsdp_schema_rejects_unknown_keys(self):
        """WS-1 round-2: FSDP block flipped extra='ignore' -> 'forbid', so a
        typo'd sharding knob raises instead of being silently swallowed."""
        assert FSDPConfigSchema().model_config["extra"] == "forbid"
        # A known field still validates.
        assert FSDPConfigSchema(sharding_strategy="no_shard").sharding_strategy == ("no_shard")
        with pytest.raises(ValidationError):
            FSDPConfigSchema(shardng_strategy="full_shard")  # typo

    def test_peft_schema_rejects_unknown_keys(self):
        assert PEFTConfigSchema().model_config["extra"] == "forbid"
        assert PEFTConfigSchema(rank=16).rank == 16
        with pytest.raises(ValidationError):
            PEFTConfigSchema(ranks=16)  # typo

    def test_fsdp_peft_schemas_are_frozen(self):
        fsdp = FSDPConfigSchema()
        peft = PEFTConfigSchema()
        with pytest.raises(ValidationError):
            fsdp.enabled = True
        with pytest.raises(ValidationError):
            peft.enabled = True

    def test_allow_idle_devices_defaults_to_armed(self):
        """#1274's refusal must be ON without the arm asking for it.

        The failure it catches is silent -- a --gpus=4 job that launched one rank
        trained happily on a quarter of its allocation while the log, the health
        report and provenance all read as healthy. A guard that had to be opted
        INTO would have been absent on exactly the arm that needed it.
        """
        assert ParallelismConfigSchema().allow_idle_devices is False
        assert ParallelismConfigSchema(allow_idle_devices=True).allow_idle_devices

    def test_allow_idle_devices_is_a_typed_knob_not_an_extra(self):
        """``extra='forbid'`` means a typo must raise rather than read as False.

        A silently-swallowed ``allow_idle_device`` would look like an opt-out that
        did not take -- the run would refuse and the operator would have no way to
        tell a rejected spelling from a working one.
        """
        assert ParallelismConfigSchema.model_config["extra"] == "forbid"
        with pytest.raises(ValidationError):
            ParallelismConfigSchema(allow_idle_device=True)  # typo
