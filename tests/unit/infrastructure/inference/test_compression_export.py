"""ModelCompressionExportService compresses, re-scores, restores and reports."""

import json

import pytest
import torch
import torch.nn as nn

from spectramr.config.schemas.compression import CompressionExportConfigSchema
from spectramr.infrastructure.inference.compression_export import (
    ARTIFACT_NAME,
    REPORT_NAME,
    ModelCompressionExportService,
)


def _cfg(**pruning):
    return CompressionExportConfigSchema.model_validate(
        {
            "enabled": True,
            "quantization": {"method": "int8_weight"},
            "pruning": pruning or {"method": "none"},
        }
    )


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 1, 3, padding=1))


def test_disabled_block_is_refused():
    with pytest.raises(ValueError, match="disabled"):
        ModelCompressionExportService(CompressionExportConfigSchema(), seed=0)


def test_swap_restores_fp32_weights():
    net = _net()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    svc = ModelCompressionExportService(_cfg(method="l1_global", amount=0.5), seed=0)
    state = svc.compress(net)
    with svc.swapped_in(net, state):
        assert (net[0].weight == 0).float().mean() >= 0.3
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k])


def test_run_writes_report_and_int_artifact(tmp_path):
    net = _net()
    x = torch.randn(2, 1, 8, 8)
    target = net(x).detach()

    def evaluate():
        with torch.no_grad():
            return {"val_mse": float(((net(x) - target) ** 2).mean())}

    svc = ModelCompressionExportService(_cfg(method="random", amount=0.4), seed=3)
    report = svc.run(net, evaluate, tmp_path)
    assert report["val_metrics_fp32"]["val_mse"] == 0.0
    assert report["val_metrics_compressed"]["val_mse"] > 0.0
    assert 1.0 < report["compression_ratio"] < 4.0
    assert report["applied"]["pruning_seed"] == 3
    on_disk = json.loads((tmp_path / REPORT_NAME).read_text())
    assert on_disk["declared"]["quantization"]["method"] == "int8_weight"
    artifact = torch.load(tmp_path / ARTIFACT_NAME)
    assert artifact["levels"]["0.weight"].dtype == torch.int8
    assert "0.bias" in artifact["dense"]
