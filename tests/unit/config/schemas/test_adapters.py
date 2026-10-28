"""Retirement of the ``adapters.pre_loss_pred`` hook (#1777)."""

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.adapters import AdaptersConfigSchema


def test_pre_loss_pred_is_refused_with_pointer() -> None:
    with pytest.raises(ValidationError, match="pre_loss_pred is retired"):
        AdaptersConfigSchema(pre_loss_pred=[{"name": "magnitude_from_complex"}])


def test_pre_loss_pred_empty_block_is_still_refused() -> None:
    with pytest.raises(ValidationError, match="pre_loss_pred is retired"):
        AdaptersConfigSchema.model_validate({"pre_loss_pred": []})


def test_remaining_hooks_load() -> None:
    cfg = AdaptersConfigSchema.model_validate(
        {"pre_model": [{"name": "slice_3d_to_2d"}], "pre_loss_target": []}
    )
    assert [hook for hook, _ in cfg.all_steps()] == ["pre_model"]
    assert not cfg.is_empty()
