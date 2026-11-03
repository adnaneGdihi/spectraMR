from spectramr.config.schemas.training.federated import (
    FederatedConfig,
    TrainingConfigFederated,
)


class TestFederatedConfig:
    def test_defaults(self):
        config = FederatedConfig()
        assert config.num_clients == 3
        assert config.local_epochs == 1
        assert config.aggregation_strategy == "fed_avg"
        assert config.client_fraction == 1.0
        assert config.num_rounds == 10

    def test_custom_values(self):
        config = FederatedConfig(
            num_clients=5, local_epochs=2, aggregation_strategy="fed_bn", num_rounds=20
        )
        assert config.num_clients == 5
        assert config.local_epochs == 2
        assert config.aggregation_strategy == "fed_bn"
        assert config.num_rounds == 20

    def test_unimplemented_rules_and_subsampling_are_refused(self):
        import pydantic
        import pytest

        with pytest.raises(pydantic.ValidationError):
            FederatedConfig(aggregation_strategy="fed_prox")
        with pytest.raises(pydantic.ValidationError, match="subsampling"):
            FederatedConfig(client_fraction=0.5)
        with pytest.raises(pydantic.ValidationError):
            FederatedConfig(fed_prox_mu=0.01)


class TestTrainingConfigFederated:
    def test_defaults(self):
        # training_mode is deprecated; create config without it
        config = TrainingConfigFederated()
        assert isinstance(config.federated_config, FederatedConfig)
        assert config.federated_config.num_clients == 3

    def test_custom_values(self):
        fed_config = FederatedConfig(num_clients=10)
        config = TrainingConfigFederated(federated_config=fed_config)
        assert config.federated_config == fed_config
        assert config.federated_config.num_clients == 10
