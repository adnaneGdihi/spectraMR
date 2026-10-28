"""Tests for the Dice-risk RCPS branch of ConformalCalibrationStrategy (B-1.7)."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.strategies.conformal_calibration_strategy import (
    ConformalCalibrationStrategy,
)


class _Identity(nn.Module):
    """Trained-model stand-in: pred = input, and load_state_dict({}) succeeds (no
    params), so a checkpoint round-trips. Lets the math tests exercise the REAL
    checkpoint-gated branch instead of bypassing it."""

    def forward(self, x, **_):
        return x


def _write_ckpt(tmp_path, model: nn.Module) -> str:
    p = tmp_path / "best.pt"
    torch.save({"generator_state_dict": model.state_dict()}, p)
    return str(p)


def _bare_strategy(dice_cfg, model: nn.Module | None = None) -> ConformalCalibrationStrategy:
    strat = object.__new__(ConformalCalibrationStrategy)
    strat.device = torch.device("cpu")
    strat._enabled_certs = ["dice_risk"]
    strat.config = types.SimpleNamespace(certification=types.SimpleNamespace(dice_risk=dice_cfg))
    strat.env = types.SimpleNamespace(generator=model if model is not None else _Identity())
    return strat


def _cfg(**over):
    base = dict(
        enabled=True,
        alpha=0.3,
        delta=0.05,
        segmenter_backend="label_dice",
        n_classes=5,
        checkpoint_path=None,
        output_artefact=None,
        # The batches below carry no subject id, so they are scored as independent
        # samples; the subject-grouped default is exercised by its own tests.
        independence_unit="sample",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_dice_risk_branch_certifies_perfect_reconstruction(tmp_path) -> None:
    # input == target with an identity model -> Dice ~1 -> risk ~0. A realistic
    # cohort (n=32) makes the Hoeffding gap small enough to certify at alpha=0.3.
    model = _Identity()
    strat = _bare_strategy(_cfg(alpha=0.3, checkpoint_path=_write_ckpt(tmp_path, model)), model)
    loader = [{"input": (x := torch.rand(4, 1, 16, 16)), "target": x} for _ in range(8)]
    reports = strat.run_calibration(loader)
    assert "r6_dice_risk" in reports
    rep = reports["r6_dice_risk"]
    assert rep["empirical_risk"] < 0.05
    assert rep["n"] == 32
    assert rep["is_certified"] is True
    # Provenance (#15): the certified checkpoint is stamped into the report.
    assert rep["checkpoint_path"].endswith("best.pt")


def test_dice_risk_branch_rejects_when_budget_too_tight(tmp_path) -> None:
    # Mismatched input/target -> nonzero Dice-risk; a tiny alpha cannot be certified.
    model = _Identity()
    strat = _bare_strategy(_cfg(alpha=0.01, checkpoint_path=_write_ckpt(tmp_path, model)), model)
    loader = [
        {"input": torch.rand(2, 1, 16, 16), "target": torch.rand(2, 1, 16, 16)} for _ in range(3)
    ]
    reports = strat.run_calibration(loader)
    assert reports["r6_dice_risk"]["is_certified"] is False


def test_dice_risk_raises_without_checkpoint() -> None:
    # ANTI-#20: certifying an untrained model is meaningless -> run_calibration must
    # raise when no checkpoint_path is declared (NOT silently certify random weights).
    strat = _bare_strategy(_cfg(alpha=0.3, checkpoint_path=None))
    loader = [{"input": torch.rand(2, 1, 16, 16), "target": torch.rand(2, 1, 16, 16)}]
    with pytest.raises(ValueError, match="checkpoint_path is required"):
        strat.run_calibration(loader)


def test_dice_risk_raises_on_missing_checkpoint_file() -> None:
    # A declared-but-absent checkpoint must fail loudly, not fall back to random init.
    strat = _bare_strategy(_cfg(alpha=0.3, checkpoint_path="/nonexistent/best.pt"))
    loader = [{"input": torch.rand(2, 1, 16, 16), "target": torch.rand(2, 1, 16, 16)}]
    with pytest.raises(FileNotFoundError):
        strat.run_calibration(loader)


def test_dice_risk_enabled_collection() -> None:
    # Drive the REAL collection logic the strategy uses (not a hand-copied loop):
    # if 'dice_risk' were dropped from _CERT_KEYS this test would now fail.
    cert = types.SimpleNamespace(
        dice_risk=_cfg(enabled=True),
        conformal=types.SimpleNamespace(enabled=False),
        chd=types.SimpleNamespace(enabled=False),
        pathology_recall=types.SimpleNamespace(enabled=False),
        pac_bayes=types.SimpleNamespace(enabled=False),
        badge=types.SimpleNamespace(enabled=False),
    )
    enabled = ConformalCalibrationStrategy._collect_enabled_certs(cert)
    assert enabled == ["dice_risk"]
    # The canonical key set still advertises dice_risk.
    assert "dice_risk" in ConformalCalibrationStrategy._CERT_KEYS


# ---------------------------------------------------------------------------
# Empirical-coverage fix (#18): the conformal certificate must MEASURE coverage
# on a held-out test split, not leave it None.
# ---------------------------------------------------------------------------


def _subject_batches(subjects_per_batch: list[list[str]]) -> list[dict]:
    """One dict batch per entry; each image is a random 8x8 pair tagged with its subject."""
    out = []
    for ids in subjects_per_batch:
        x = torch.rand(len(ids), 1, 8, 8)
        out.append({"input": x, "target": x + 0.1 * torch.rand_like(x), "subject_id": ids})
    return out


def _conformal_strategy(seed: int = 42, **cert) -> ConformalCalibrationStrategy:
    strat = object.__new__(ConformalCalibrationStrategy)
    strat.device = torch.device("cpu")
    strat._enabled_certs = list(cert)
    strat.config = types.SimpleNamespace(
        run=types.SimpleNamespace(seed=seed),
        certification=types.SimpleNamespace(**cert),
    )
    strat.env = types.SimpleNamespace(generator=_Identity())
    return strat


def test_subject_split_keeps_every_subject_on_one_side() -> None:
    batches = _subject_batches([["a", "a", "b"], ["b", "c", "c"], ["d", "d", "e"]])
    calib, test, info = _conformal_strategy()._subject_split(batches, seed=7)
    cal_ids = {sid for b in calib for sid in b["subject_id"]}
    test_ids = {sid for b in test for sid in b["subject_id"]}
    assert cal_ids and test_ids
    assert cal_ids.isdisjoint(test_ids)
    assert cal_ids | test_ids == {"a", "b", "c", "d", "e"}
    assert info["split_unit"] == "subject" and info["split_seed"] == 7


def test_subject_split_is_seeded_from_run_seed() -> None:
    batches = _subject_batches([[f"s{i}", f"s{i}"] for i in range(8)])
    strat = _conformal_strategy()
    first = strat._subject_split(batches, seed=3)[0]
    again = strat._subject_split(batches, seed=3)[0]
    assert [b["subject_id"] for b in first] == [b["subject_id"] for b in again]


def test_subject_split_refuses_fewer_than_two_subjects() -> None:
    with pytest.raises(ValueError, match="at least 2 subjects"):
        _conformal_strategy()._subject_split(_subject_batches([["a", "a"]]), seed=1)


def test_cap_images_keeps_exactly_n_and_raises_when_fewer() -> None:
    from spectramr.infrastructure.training.strategies.conformal_calibration_strategy import (
        _cap_images,
    )

    batches = _subject_batches([["a", "a", "a"], ["b", "b", "b"]])
    kept, n = _cap_images(batches, 4)
    assert n == 4
    assert sum(len(b["subject_id"]) for b in kept) == 4
    with pytest.raises(ValueError, match="n_calibration=7 but the calibration half holds 6"):
        _cap_images(batches, 7)


def test_calibration_input_target_unpacks_dict_and_tuple() -> None:
    from spectramr.infrastructure.training.strategies.conformal_calibration_strategy import (
        _calibration_input_target,
    )

    assert _calibration_input_target({"input": "X", "target": "Y"}) == ("X", "Y")
    assert _calibration_input_target(("X", "Y")) == ("X", "Y")
    with pytest.raises(ValueError, match="dict with input/target"):
        _calibration_input_target(object())


def test_runner_measures_coverage_only_with_test_split() -> None:
    """End-to-end: empirical_coverage is None without a test loader (the old
    bug) and a real number once a disjoint test split is supplied."""
    from spectramr.infrastructure.calibration import (
        ConformalCalibrationRunner,
        ConformalCalibrator,
    )
    from spectramr.infrastructure.training.strategies.conformal_calibration_strategy import (
        _calibration_input_target,
    )

    torch.manual_seed(0)

    def score(pred, target):
        return (pred - target).abs().flatten(1).amax(1)

    batches = [
        {"input": torch.randn(2, 1, 8, 8), "target": torch.randn(2, 1, 8, 8)} for _ in range(6)
    ]
    calib, test = batches[:3], batches[3:]
    runner = ConformalCalibrationRunner(
        model=nn.Identity(),
        calibrator=ConformalCalibrator(score_fn=score, alpha=0.1),
        device="cpu",
        unpack_batch=_calibration_input_target,
    )
    assert runner.run(calib).empirical_coverage is None
    cov = runner.run(calib, test).empirical_coverage
    assert cov is not None and 0.0 <= cov <= 1.0


def test_run_calibration_passes_held_out_test_split() -> None:
    """Source-pin: the conformal certificate runs the calibrator on a held-out
    split so empirical_coverage is produced (guards against a regression back to
    the single-arg ``runner.run(dataloader)`` that left coverage None)."""
    import pathlib

    src = pathlib.Path(
        "src/spectramr/infrastructure/training/strategies/conformal_calibration_strategy.py"
    ).read_text(encoding="utf-8")
    assert "runner.run(calib_batches, test_all)" in src
    assert "self._subject_split(batches, int(self.config.run.seed))" in src


# ── #1707: one Hoeffding draw per INDEPENDENT unit ─────────────────────────────


def test_subject_unit_counts_subjects_not_slices(tmp_path) -> None:
    """Planted violation: 3 subjects x 8 slices used to certify with n = 24."""
    model = _Identity()
    strat = _bare_strategy(
        _cfg(alpha=0.3, independence_unit="subject", checkpoint_path=_write_ckpt(tmp_path, model)),
        model,
    )
    loader = []
    for sid in ("s1", "s2", "s3"):
        for _ in range(2):
            x = torch.rand(4, 1, 16, 16)
            loader.append({"input": x, "target": x, "subject_id": [sid] * 4})
    rep = strat.run_calibration(loader)["r6_dice_risk"]
    assert rep["n"] == 3 and rep["n_samples_scored"] == 24
    assert rep["independence_unit"] == "subject"


def test_subject_unit_uses_the_per_subject_mean_risk(tmp_path) -> None:
    """Two subjects: one perfect (risk ~0), one scored on a mismatched pair; the empirical
    risk is the mean of the two subject means, not of the 8 slices."""
    model = _Identity()
    strat = _bare_strategy(
        _cfg(alpha=0.3, independence_unit="subject", checkpoint_path=_write_ckpt(tmp_path, model)),
        model,
    )
    good = torch.rand(4, 1, 16, 16)
    bad_in, bad_tgt = torch.rand(4, 1, 16, 16), torch.rand(4, 1, 16, 16)
    loader = [
        {"input": good, "target": good, "subject_id": ["a"] * 4},
        {"input": bad_in, "target": bad_tgt, "subject_id": ["b"] * 4},
    ]
    rep = strat.run_calibration(loader)["r6_dice_risk"]
    assert rep["n"] == 2
    assert 0.0 < rep["empirical_risk"] < 1.0


def test_subject_unit_refuses_a_batch_without_subject_ids(tmp_path) -> None:
    """No silent fallback to slice counting (the inflation #1707 describes)."""
    model = _Identity()
    strat = _bare_strategy(
        _cfg(independence_unit="subject", checkpoint_path=_write_ckpt(tmp_path, model)), model
    )
    x = torch.rand(4, 1, 16, 16)
    with pytest.raises(ValueError, match="subject_id"):
        strat.run_calibration([{"input": x, "target": x}])


def test_sample_unit_keeps_the_old_count_and_says_so(tmp_path) -> None:
    model = _Identity()
    strat = _bare_strategy(_cfg(checkpoint_path=_write_ckpt(tmp_path, model)), model)
    x = torch.rand(4, 1, 16, 16)
    rep = strat.run_calibration([{"input": x, "target": x, "subject_id": ["a"] * 4}])[
        "r6_dice_risk"
    ]
    assert rep["n"] == 4 and rep["independence_unit"] == "sample" and rep["n_samples_scored"] == 4


class _GeometryModel(nn.Module):
    """Stand-in for a geometry-plane generator: raises without ``mask=``, records it."""

    kspace_geometry_channels = 6

    def __init__(self) -> None:
        super().__init__()
        self.masks: list[torch.Tensor] = []

    def forward(self, x, mask=None):
        if mask is None:
            raise ValueError("forward needs mask=")
        self.masks.append(mask)
        return x


@pytest.mark.parametrize("as_dict", [True, False])
def test_calibration_pair_gives_a_geometry_model_the_measurement_support(as_dict) -> None:
    """The dice/CHD paths called the kspace_cold_diffusion generator without a mask."""
    from spectramr.models.blocks.kspace_geometry_encoding import measurement_support

    y = torch.zeros(1, 2, 8, 8)
    y[..., 1::3] = 1.0
    batch = {"input": y, "target": y.clone()} if as_dict else (y, y.clone())
    model = _GeometryModel()
    pred, target = ConformalCalibrationStrategy._unpack_calibration_pair(batch, model)
    assert pred is y and target is not None
    torch.testing.assert_close(model.masks[0], measurement_support(y))


def test_calibration_pair_passes_no_mask_to_a_model_without_geometry_planes() -> None:
    seen: list[dict] = []

    class _Bare(nn.Module):
        def forward(self, x, **kwargs):
            seen.append(kwargs)
            return x

    y = torch.ones(1, 2, 4, 4)
    ConformalCalibrationStrategy._unpack_calibration_pair((y, y), _Bare())
    assert seen == [{}]


# ── Planted violations for the certificate branches (#3041) ────────────────────


def test_chd_branch_writes_a_stamped_artefact(tmp_path) -> None:
    """CHD fits on the calibration half and measures FPR on the disjoint test half."""
    out = tmp_path / "chd.json"
    chd = types.SimpleNamespace(
        beta=0.2,
        delta=0.05,
        n_calibration=4,
        foreground_threshold=None,
        output_artefact=str(out),
    )
    strat = _conformal_strategy(chd=chd)
    loader = _subject_batches([["a", "a"], ["b", "b"], ["c", "c"], ["d", "d"]])
    torch.manual_seed(0)
    reports = strat.run_calibration(loader)
    payload = reports["r2_chd"]
    assert out.exists()
    assert payload["n_calibration_images"] == 4
    assert payload["n_calibration_requested"] == 4
    assert payload["split_unit"] == "subject"
    assert 0.0 <= payload["empirical_fpr"] <= 1.0
    assert payload["score_fn"] == "absolute_residual"


def test_chd_cap_raises_when_the_calibration_half_is_too_small() -> None:
    chd = types.SimpleNamespace(
        beta=0.2, delta=0.05, n_calibration=50, foreground_threshold=None, output_artefact=None
    )
    strat = _conformal_strategy(chd=chd)
    loader = _subject_batches([["a", "a"], ["b", "b"], ["c", "c"], ["d", "d"]])
    with pytest.raises(ValueError, match="n_calibration=50"):
        strat.run_calibration(loader)


@pytest.mark.parametrize(
    ("cert_name", "cert", "message"),
    [
        (
            "pathology_recall",
            types.SimpleNamespace(
                classes=["infarct"], delta=0.05, target_recall=0.9, output_artefact=None
            ),
            "lesion detector",
        ),
        (
            "pac_bayes",
            types.SimpleNamespace(
                delta=0.05, tv_tolerance_tau=0.05, loss_bound_M=1.0, output_artefact=None
            ),
            "KL",
        ),
    ],
)
def test_uncomputable_certificates_refuse_instead_of_faking_a_result(cert_name, cert, message):
    strat = _conformal_strategy(**{cert_name: cert})
    with pytest.raises(NotImplementedError, match=message):
        strat.run_calibration(_subject_batches([["a", "a"], ["b", "b"]]))


def test_subject_split_names_the_loader_when_subject_id_is_missing() -> None:
    """Planted: a val loader without subject_id must fail with a conformal-specific message."""
    batches = [{"input": torch.rand(2, 1, 8, 8), "target": torch.rand(2, 1, 8, 8)}]
    with pytest.raises(ValueError, match="M4Raw k-space val loader must emit it"):
        _conformal_strategy()._subject_split(batches, seed=1)
