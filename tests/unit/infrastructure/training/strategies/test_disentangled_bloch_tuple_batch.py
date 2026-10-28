"""Regression: _synthesize_via_bloch must tolerate the (input, target) TUPLE
that validation_step passes as ``batch``.

2026-05-31 audit: for ``output_mode: tissue_params`` arms, validation_step
builds ``batch = (input_batch, target_batch)`` and hands it to
``_synthesize_via_bloch`` -> ``_synthesize_via_bloch_fp32`` does
``batch.get("TR", ...)`` -> ``AttributeError: 'tuple' object has no attribute
'get'``. The wrapper now coerces a non-dict batch to ``{}`` (defaults apply).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.disentangled_strategy import (
    DisentangledTrainingStrategy,
)


class TestResolveAcqParamNoSilentDefault:
    """F4: acquisition timing params must come from the model prediction or the
    batch — never a hardcoded GRE default (pitfall #9, silent fallback)."""

    def test_prefers_model_prediction(self):
        v = DisentangledTrainingStrategy._resolve_acq_param(
            {"predicted_tr": torch.tensor(1500.0)}, {"TR": 8.0}, "predicted_tr", "TR"
        )
        assert float(v) == 1500.0

    def test_falls_back_to_batch_metadata(self):
        v = DisentangledTrainingStrategy._resolve_acq_param(
            {}, {"TR": 2000.0}, "predicted_tr", "TR"
        )
        assert float(v) == 2000.0

    def test_raises_when_neither_source_present(self):
        # The old code silently returned 8.0 here; it must now fail loud.
        with pytest.raises(ValueError, match="predicted_tr"):
            DisentangledTrainingStrategy._resolve_acq_param(
                {}, {}, "predicted_tr", "TR"
            )


def test_synthesize_via_bloch_passes_the_real_batch_through(monkeypatch):
    """Coercing a TrainingBatch to ``{}`` rendered every sample as GRE (#2611)."""
    from spectramr.data.batch_types import BatchAdapter

    s = object.__new__(DisentangledTrainingStrategy)
    received = {}

    def _fake_fp32(self, tissue_params, batch, source_key):
        received["batch"] = batch
        return torch.zeros(1)

    monkeypatch.setattr(DisentangledTrainingStrategy, "_synthesize_via_bloch_fp32", _fake_fp32)
    x = torch.randn(1, 1, 4, 4)
    batch = BatchAdapter.from_dict({"input": x, "target": x, "contrast": ["T1"]})
    s._synthesize_via_bloch({}, batch, "a")
    assert received["batch"] is batch


def _bare_strategy(sequence_map: dict[str, int]) -> DisentangledTrainingStrategy:
    s = object.__new__(DisentangledTrainingStrategy)
    s._sequence_map = sequence_map
    s._fno_epg = None
    s._in_channels = 1
    s._dipole_pad_z = 0
    return s


def _tissue(n: int, *, b1: bool) -> dict:
    t = {k: torch.full((n, 1, 4, 4), v) for k, v in (("rho", 0.8), ("t1", 800.0), ("t2", 90.0))}
    t.update({f"predicted_{p}": torch.full((n, 1, 1, 1), v) for p, v in
              (("tr", 500.0), ("te", 20.0), ("ti", 400.0), ("alpha", 90.0))})
    if b1:
        t["b1_plus"] = torch.ones(n, 1, 4, 4)
    return t


def test_bloch_layer_receives_per_sample_codes_from_the_batch():
    """A GRE row keeps the batch off the EPG paths, so it reaches the Bloch layer."""
    from spectramr.data.batch_types import BatchAdapter

    s = _bare_strategy({"T1": 2, "FLAIR": 1})
    seen = {}

    def _layer(tissue, meta):
        seen["meta"] = meta
        return torch.zeros(2, 1, 4, 4)

    s._bloch_layer = _layer
    x = torch.zeros(2, 1, 4, 4)
    batch = BatchAdapter.from_dict({"input": x, "target": x, "contrast": ["T1", "FLAIR"]})
    s._synthesize_via_bloch(_tissue(2, b1=False), batch, "a")
    assert seen["meta"]["contrast_type"].view(-1).tolist() == [2, 1]


def test_epg_path_applies_inversion_recovery_to_ir_rows_only():
    """All-SE/IR batches take the EPG path; the IR prep must follow each row's family."""
    s = _bare_strategy({"T2": 0, "FLAIR": 1})
    x = torch.zeros(2, 1, 4, 4)
    tissue = _tissue(2, b1=True)
    # Two T1 values: the inversion changes relative contrast, which per-sample
    # normalisation keeps (a uniform image would normalise any constant away).
    tissue["t1"][..., :2] = 3000.0
    out_se = s._synthesize_via_bloch(dict(tissue), {"input": x, "contrast": ["T2", "T2"]}, "a")
    out_mix = s._synthesize_via_bloch(dict(tissue), {"input": x, "contrast": ["T2", "FLAIR"]}, "a")
    assert torch.allclose(out_se[0], out_mix[0])  # the SE row is unchanged by its neighbour
    assert not torch.allclose(out_se[1], out_mix[1])  # the IR row carries the inversion prep


def test_the_bloch_layer_path_applies_the_bias_field_once():
    """MultiPhysicsBlochLayer multiplies by bias_field itself; the strategy must not again."""
    s = _bare_strategy({"T1": 2})
    s._bloch_layer = lambda tissue, meta: torch.ones(1, 1, 4, 4)  # stands in for the layer
    tissue = _tissue(1, b1=False)
    tissue["bias_field"] = torch.full((1, 1, 4, 4), 2.0)
    out = s._synthesize_via_bloch(tissue, {"contrast": ["T1"]}, "a")
    assert torch.allclose(out, torch.ones(1, 1, 4, 4))  # 1 * 2 - 1: the layer's output, untouched


def test_a_tuple_batch_has_no_sequence_family_and_raises():
    s = _bare_strategy({"T1": 2})
    with pytest.raises(ValueError, match="neither 'contrast_type' nor 'contrast'"):
        s._synthesize_via_bloch(_tissue(1, b1=False), (torch.zeros(1), torch.zeros(1)), "a")


def test_validation_step_declares_batch_data():
    """train.py forwards the real batch to a Pattern-A validation_step only on declaration."""
    import inspect

    assert "batch_data" in inspect.signature(DisentangledTrainingStrategy.validation_step).parameters


def test_fno_epg_path_backpropagates_a_predicted_b1_plus_map():
    """`use_fno_epg` arms (experiment 51) crashed at the first backward: the predicted B1+
    map reached the surrogate, which edited a tensor autograd had saved (sabine, 2026-10-08)."""
    from spectramr.infrastructure.physics.fno_epg import FNOEPGSurrogate

    torch.manual_seed(0)
    s = _bare_strategy({"T2": 0, "FLAIR": 1})
    s._fno_epg = FNOEPGSurrogate(tissue_channels=4, hidden_width=8, n_layers=1, max_etl=16)
    tissue = _tissue(2, b1=True)
    leaves = {k: tissue[k].requires_grad_(True) for k in ("rho", "t1", "t2", "b1_plus")}
    x = torch.zeros(2, 1, 4, 4)

    out = s._synthesize_via_bloch(tissue, {"input": x, "contrast": ["T2", "FLAIR"]}, "a")
    out.sum().backward()

    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in leaves.values())
