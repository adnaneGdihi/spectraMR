import pytest
from pydantic import ValidationError

from spectramr.config.schemas.augmentation import AugmentationConfigSchema


class TestAugmentationConfigSchema:
    def test_defaults(self):
        schema = AugmentationConfigSchema()
        assert schema.enabled is False
        assert schema.probability == 0.5
        assert schema.enable_rotation is False
        assert schema.rotation_range == (-15, 15)

    def test_custom_values(self):
        schema = AugmentationConfigSchema(
            enabled=True,
            probability=0.8,
            enable_rotation=True,
            rotation_range=(-30, 30),
            enable_rician_noise=True,
            rician_noise_level=0.1,
        )
        assert schema.enabled is True
        assert schema.probability == 0.8
        assert schema.enable_rotation is True
        assert schema.rotation_range == (-30, 30)
        assert schema.enable_rician_noise is True
        assert schema.rician_noise_level == 0.1

    def test_validation(self):
        with pytest.raises(ValidationError):
            AugmentationConfigSchema(probability=1.5)
        with pytest.raises(ValidationError):
            AugmentationConfigSchema(probability=-0.1)
        with pytest.raises(ValidationError):
            AugmentationConfigSchema(prob_rotate=1.5)
        with pytest.raises(ValidationError):
            AugmentationConfigSchema(elastic_alpha=-1)

    def test_extra_fields(self):
        with pytest.raises(ValidationError):
            AugmentationConfigSchema(extra_field="value")
