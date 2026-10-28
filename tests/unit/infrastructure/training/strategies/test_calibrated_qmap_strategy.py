"""Tests for CalibratedQMapStrategy (exp 32, gain-calibrated maps from noise-unit M4Raw)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.calibrated_qmap_strategy import (
    CalibratedQMapStrategy,
)
from spectramr.models.generators.bloch_parameter_bottleneck import (
    BlochParameterBottleneckGenerator,
)

# Census protocols: T1 SE (ETL 1, echo at TE), T2 TSE, FLAIR IR-TSE.
_PROTO = {
    "acq_tr": [500.0, 5500.0, 7500.0],
    "acq_te": [18.4, 128.0, 98.0],
    "acq_ti": [0.0, 0.0, 1655.0],
    "acq_esp": [18.4, 16.0, 16.3],
    "acq_etl": [1, 15, 11],
    "acq_refocus_deg": [180.0, 160.0, 160.0],
    "sequence_family": [0, 0, 1],
}


def _batch(b: int = 2) -> dict:
    torch.manual_seed(0)
    x = torch.rand(b, 3, 32, 32) * 50.0
    x[..., :20, :] = 1.0  # air, most of the slice
    out = {"input": x, "target": x}
    out.update({k: torch.tensor([v] * b) for k, v in _PROTO.items()})
    return out


def _strategy(
    mode: str = "calibrated", gains: dict | None = None, model=None, **extra
) -> CalibratedQMapStrategy:
    s = CalibratedQMapStrategy.__new__(CalibratedQMapStrategy)
    s.device = torch.device("cpu")
    model = model or BlochParameterBottleneckGenerator(in_channels=3, base_channels=8)
    mk = {"gain_mode": mode, **extra}
    if gains is not None:
        mk["series_gain"] = gains
    s.config = SimpleNamespace(model=SimpleNamespace(model_kwargs=mk))
    s.env = SimpleNamespace(generator=model, opt_g=torch.optim.Adam(model.parameters()))
    s._configure_calibrated_qmap()
    return s


_GAINS = {"T1": 1.0, "T2": 1.8, "FLAIR": 0.6}
_LOO_KEYS = {"val_psnr_loo", "val_psnr_loo_t1", "val_psnr_loo_t2", "val_psnr_loo_flair"}


def test_calibrated_loss_is_finite_and_trains_the_maps_and_the_global_gain() -> None:
    s, batch = _strategy(gains=_GAINS), _batch()
    out = s._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    out["g_total_loss"].backward()
    assert torch.isfinite(out["g_total_loss"])
    assert next(s.env.generator.parameters()).grad is not None
    assert s._log_global_gain.grad is not None


def test_the_global_gain_reaches_the_optimizer() -> None:
    s = _strategy(gains=_GAINS)
    groups = [p for g in s.env.opt_g.param_groups for p in g["params"]]
    assert any(p is s._log_global_gain for p in groups)


def test_projected_mode_needs_no_gains_and_learns_no_global_gain() -> None:
    s, batch = _strategy("projected"), _batch()
    out = s._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    assert torch.isfinite(out["g_total_loss"]) and s._log_global_gain is None


def test_calibrated_mode_requires_all_three_gains() -> None:
    with pytest.raises(ValueError, match="series_gain"):
        _strategy(gains={"T1": 1.0, "T2": 2.0})


def test_a_non_positive_gain_raises() -> None:
    with pytest.raises(ValueError, match="> 0"):
        _strategy(gains={"T1": 1.0, "T2": 0.0, "FLAIR": 0.5})


def test_an_unknown_gain_mode_raises() -> None:
    with pytest.raises(ValueError, match="gain_mode"):
        _strategy("bounded")


def test_validation_emits_fidelity_per_batch_and_spread_once_per_pass() -> None:
    s, batch = _strategy(gains=_GAINS), _batch()
    batch["subject_id"] = ["a", "b"]
    s._psnr = lambda pred, target: torch.tensor(30.0)
    s.reset_validation_metrics()
    metrics = s.validation_step(batch["input"], batch["target"], batch_data=batch)
    # Per-batch readings only; a per-batch CV would be a within-subject spread.
    assert set(metrics) == {"val_psnr", *_LOO_KEYS}
    summary = s.finalize_validation()
    assert {"val_median_t1_wm", "val_median_t2_wm", "val_cv_t1_wm", "val_cv_t2_wm"} <= set(summary)


def test_the_spread_is_across_subjects_not_slices() -> None:
    """Each subject contributes the median of its slices; the CV is over subjects."""
    s = _strategy(gains=_GAINS)
    s.reset_validation_metrics()
    s._record_white_matter(["a", "a", "b"], torch.tensor([600.0, 620.0, 800.0]), torch.ones(3))
    s._record_white_matter(["b"], torch.tensor([820.0]), torch.ones(1))
    out = s.finalize_validation()
    subject_t1 = torch.tensor([610.0, 810.0])  # subject a: median(600, 620); b: median(800, 820)
    assert out["val_median_t1_wm"] == pytest.approx(710.0)
    assert out["val_cv_t1_wm"] == pytest.approx(
        float(subject_t1.std(unbiased=False) / subject_t1.mean())
    )


def test_a_batch_without_subject_ids_raises() -> None:
    s, batch = _strategy(gains=_GAINS), _batch()
    s._psnr = lambda pred, target: torch.tensor(30.0)
    s.reset_validation_metrics()
    with pytest.raises(ValueError, match="subject_id"):
        s.validation_step(batch["input"], batch["target"], batch_data=batch)


def test_calibrated_mode_is_refused_under_a_process_group(monkeypatch) -> None:
    """The gain is outside the wrapped generator, so no rank would reduce its gradient."""
    monkeypatch.setattr(
        "spectramr.infrastructure.distributed.distributed_training.RankUtility.get_world_size",
        staticmethod(lambda: 2),
    )
    with pytest.raises(ValueError, match="single-process"):
        _strategy(gains=_GAINS)
    assert _strategy("projected") is not None  # the control has no such parameter


def test_a_restored_gain_is_not_reseeded_from_the_first_batch() -> None:
    donor = _strategy(gains=_GAINS)
    with torch.no_grad():
        donor._log_global_gain.fill_(2.0)
    state = donor.strategy_state_dict()
    assert "_log_global_gain" in state["params"]

    resumed, batch = _strategy(gains=_GAINS), _batch()
    resumed.load_strategy_state_dict(state)
    resumed._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    assert float(resumed._log_global_gain) == pytest.approx(2.0)


def test_a_validation_protocol_that_differs_from_training_raises() -> None:
    s, batch = _strategy(gains=_GAINS), _batch()
    s._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    other = _batch()
    other["acq_tr"] = torch.tensor([[600.0, 5500.0, 7500.0]] * 2)
    other["subject_id"] = ["a", "b"]
    s._psnr = lambda pred, target: torch.tensor(30.0)
    with pytest.raises(ValueError, match="differs from training"):
        s.validation_step(other["input"], other["target"], batch_data=other)


def test_the_global_gain_starts_at_the_data_scale_without_weight_decay() -> None:
    """Bounded apparent PD against noise-unit targets: starting at 1 left the gain to
    crawl up for 1e4+ steps while the maps bent to make up the scale."""
    s, batch = _strategy(gains=_GAINS), _batch()
    batch["target"] = batch["target"] * 40.0
    s._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    assert float(s._log_global_gain.detach().exp()) > 5.0
    group = next(g for g in s.env.opt_g.param_groups if g.get("name") == "global_gain")
    assert group["weight_decay"] == 0.0


def test_train_step_accepts_the_three_series_target() -> None:
    """The target is the acquired T1/T2/FLAIR the strategy renders, not the model output.

    REGRESSION (final review): the base width guard raised "[DomainMismatch] Model outputs
    1 channels, but target dataset provided 3" on step 1. Drives one real train_step.
    """
    import types

    s, batch = _strategy(gains=_GAINS), _batch()
    s.config = types.SimpleNamespace(
        model=types.SimpleNamespace(
            model_type="bloch_parameter_bottleneck",
            in_channels=3,
            out_channels=1,
            model_kwargs=s.config.model.model_kwargs,
        ),
        training=types.SimpleNamespace(strategy_class="calibrated_qmap"),
    )
    s.adapter_chains = {}
    s._prepare_model_input = lambda x: x  # type: ignore[method-assign]
    s.env.discriminator = None
    s.train_step(
        batch, epoch=0, input_batch=batch["input"], target_batch=batch["target"], iteration=1
    )


def test_a_depth_other_than_one_is_refused() -> None:
    s, batch = _strategy(gains=_GAINS), _batch()
    volume = batch["input"].unsqueeze(-1).repeat(1, 1, 1, 1, 2)  # (B, 3, H, W, 2)
    with pytest.raises(ValueError, match="one slice"):
        s._compute_losses_impl(volume, volume, epoch=0, batch=batch)


def test_each_series_is_scored_on_its_own_signal_voxels() -> None:
    """Planted: a voxel dark in T1 and bright in T2/FLAIR (CSF) must be in the T2 mask."""
    y = torch.zeros(1, 3, 4, 4)
    y[:, 0, :2] = 20.0  # T1 signal in the top half only
    y[:, 1] = 20.0  # T2 signal everywhere, including where T1 is dark
    y[:, 2, 2:] = 20.0  # FLAIR signal in the bottom half only
    t1, t2, flair = CalibratedQMapStrategy._signal_masks(y)

    assert t1[0, 0, :2].all() and not t1[0, 0, 2:].any()
    assert t2.all()
    assert flair[0, 0, 2:].all() and not flair[0, 0, :2].any()
    assert (t2 & ~t1).sum() != 0  # the T2 mask has voxels a T1-only mask would drop


def test_the_loss_changes_when_only_a_t2_voxel_changes_where_t1_is_dark() -> None:
    s, batch = _strategy("projected"), _batch()
    base = s._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    y = batch["target"].clone()
    y[:, 1, :20, :] = 40.0  # T2 only, in the air rows that every T1 mask excludes
    changed = s._compute_losses_impl(batch["input"], y, epoch=0, batch=batch)
    assert not torch.allclose(base["g_total_loss"], changed["g_total_loss"])


def _spy_inputs(s: CalibratedQMapStrategy) -> list:
    seen = []
    real = s.env.generator.predict_parameters
    s.env.generator.predict_parameters = lambda x: seen.append(x.clone()) or real(x)
    return seen


def test_without_dropout_the_tissue_unet_sees_the_whole_triplet() -> None:
    # The #2622 control is unchanged unless an arm asks for contrast dropout.
    s, batch = _strategy("projected"), _batch()
    seen = _spy_inputs(s)
    s._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    assert torch.equal(seen[0], batch["input"])


def test_with_dropout_a_hidden_series_reaches_the_tissue_unet_as_zeros() -> None:
    torch.manual_seed(3)
    s, batch = _strategy("projected", contrast_dropout=0.6), _batch(64)
    seen = _spy_inputs(s)
    s._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    hidden = seen[0].flatten(2).abs().sum(2) == 0
    assert hidden.any() and not hidden.all(1).any()  # some hidden, never all three


def test_a_dropout_outside_0_1_is_refused() -> None:
    with pytest.raises(ValueError, match="contrast_dropout"):
        _strategy("projected", contrast_dropout=1.0)


def _unrolled():
    from spectramr.models.generators.unrolled_tissue_net import UnrolledTissueNet

    torch.manual_seed(0)
    return UnrolledTissueNet(base_channels=8, iterations=2)


def test_the_unrolled_net_trains_through_the_strategy_with_dropout() -> None:
    s, batch = _strategy("projected", model=_unrolled(), contrast_dropout=0.3), _batch()
    s.env.generator.train()
    out = s._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    out["g_total_loss"].backward()
    assert torch.isfinite(out["g_total_loss"])
    assert s.env.generator.init_net[1].weight.grad is not None


def test_the_unrolled_net_reports_leave_one_out_in_validation() -> None:
    s, batch = _strategy("projected", model=_unrolled()), _batch()
    batch["subject_id"] = ["a", "b"]
    s._psnr = lambda pred, target: torch.tensor(30.0)
    s.reset_validation_metrics()
    metrics = s.validation_step(batch["input"], batch["target"], batch_data=batch)
    assert set(metrics) == {"val_psnr", *_LOO_KEYS}


_PRIOR = {
    "weight": 1.0,
    "T2": {"median": 4.2, "std_log": 0.16},
    "FLAIR": {"median": 0.51, "std_log": 0.29},
}


def test_the_gauge_prior_adds_to_the_projected_loss() -> None:
    batch = _batch()
    without = _strategy("projected")
    with_prior = _strategy("projected", gain_ratio_prior=_PRIOR)
    with_prior.env.generator.load_state_dict(without.env.generator.state_dict())
    a = without._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    b = with_prior._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    assert float(b["g_total_loss"]) > float(a["g_total_loss"])


def test_with_the_prior_validation_scores_a_label_free_gain_and_conformal_intervals() -> None:
    s = _strategy("projected", gain_ratio_prior=_PRIOR)
    s._psnr = lambda pred, target: torch.tensor(30.0)
    s.reset_validation_metrics()
    for subject in ("a", "b", "c", "d"):
        batch = _batch()
        batch["subject_id"] = [subject + "1", subject + "2"]
        metrics = s.validation_step(batch["input"], batch["target"], batch_data=batch)
    prior_keys = {"val_psnr_loo_prior_gain"} | {
        f"val_psnr_loo_prior_gain_{c}" for c in ("t1", "t2", "flair")
    }
    assert set(metrics) == {"val_psnr", *_LOO_KEYS, *prior_keys}
    summary = s.finalize_validation()
    assert any(k.startswith("val_conformal_q_") for k in summary)


def test_a_repetition_split_batch_adds_the_reference_free_risk() -> None:
    s, batch = _strategy("projected"), _batch()
    batch["subject_id"] = ["a", "b"]
    batch["noise_power"] = torch.tensor([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]])
    s._psnr = lambda pred, target: torch.tensor(30.0)
    s.reset_validation_metrics()
    metrics = s.validation_step(batch["input"], batch["target"], batch_data=batch)
    assert {"val_rep_split_risk", "val_rep_split_bias"} <= set(metrics)


def test_the_physics_free_control_trains_and_validates_without_maps() -> None:
    from spectramr.models.generators.masked_contrast_unet import MaskedContrastUNet

    torch.manual_seed(0)
    model = MaskedContrastUNet(base_channels=8)
    s, batch = _strategy("projected", model=model, contrast_dropout=0.3), _batch()
    out = s._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    out["g_total_loss"].backward()
    assert model.head.weight.grad is not None
    batch["subject_id"] = ["a", "b"]
    s._psnr = lambda pred, target: torch.tensor(30.0)
    s.reset_validation_metrics()
    assert set(s.validation_step(batch["input"], batch["target"], batch_data=batch)) == {
        "val_psnr",
        *_LOO_KEYS,
    }
    assert not any(k.endswith("_wm") for k in s.finalize_validation())  # no tissue maps


def test_the_strategy_declares_that_it_owns_its_objective() -> None:
    # The q-map arms declare no losses: block, since the gain-calibrated render loss is
    # built here. Without the declaration the loss builder refused every q-map arm at
    # build time (sabine job 17889904, B2 and C1).
    from spectramr.infrastructure.training.strategies.loss_folding import (
        declares_inline_objective,
    )

    assert declares_inline_objective(CalibratedQMapStrategy)


def test_the_gauge_prior_is_refused_in_calibrated_mode() -> None:
    with pytest.raises(ValueError, match="projected"):
        _strategy(gains=_GAINS, gain_ratio_prior=_PRIOR)
