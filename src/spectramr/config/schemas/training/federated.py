"""Federated Learning configuration schema.

This module defines TrainingConfigFederated for federated learning paradigms.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .base import BaseTrainingConfigSchema

#: The rules FederatedRoundController implements; nothing reads FedProx/FedOpt/SCAFFOLD/FedNova
#: knobs, so they are not advertised.
VALID_FED_AGGREGATIONS = ("fed_avg", "fed_bn")


class FederatedConfig(BaseModel):
    """Federated Learning specific configuration."""

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    num_clients: int = Field(
        default=3,
        ge=2,
        description="Number of federated clients (must be >= 2 — single-client is not federated)",
    )
    local_epochs: int = Field(
        default=1,
        ge=1,
        le=1000,
        description="Number of local epochs per round",
    )
    aggregation_strategy: Literal["fed_avg", "fed_bn"] = Field(
        default="fed_avg",
        description=(
            "fed_avg: n_k-weighted mean of every parameter and buffer. fed_bn: the same, "
            "but each site keeps its own normalisation layers and is validated with them."
        ),
    )
    client_fraction: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Fraction of clients per round. Only 1.0: client subsampling is not implemented.",
    )
    num_rounds: int = Field(
        default=10,
        ge=1,
        le=100000,
        description="Number of federated rounds; one round is one training epoch, so it must equal training.epochs.",
    )

    @field_validator("aggregation_strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        if v not in VALID_FED_AGGREGATIONS:
            raise ValueError(
                f"aggregation_strategy must be one of {VALID_FED_AGGREGATIONS}, got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _enforce_min_active_clients(self) -> "FederatedConfig":
        if self.client_fraction != 1.0:
            raise ValueError(
                f"client_fraction={self.client_fraction}: client subsampling is not "
                "implemented; every client trains every round (1.0)"
            )
        return self


class TrainingConfigFederated(BaseTrainingConfigSchema):
    """Federated Learning training configuration.

    Enforces federated-specific configurations.
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    federated_config: FederatedConfig = Field(
        default_factory=FederatedConfig,
        description="Federated learning configuration",
    )
