"""Invariants for the two-arm sharpness ablation.

The cohort exists because `experiment_11_attention_none`'s reconstructions carry
0.078-0.65x of the target's radial spectral power above 0.4 normalised frequency, while
low and mid bands match at 0.99-1.05x (measured by
``scripts/diagnostics/spectral_sharpness.py``). The blur is a high-frequency deficit, and
every term in that recipe is L1-family -- an objective whose minimiser is the conditional
median, hence smooth wherever the measurement does not determine the answer.

Two properties have to hold or the ablation means nothing:

* **The knob is the only difference.** Both arms must agree on backbone, mask, data
  consistency, optimizer, EMA, schedule *and the checkpoint-selection metric*; the sole
  delta is whether ``hfen`` appears in ``losses.image_losses`` (pitfall #17).
* **The knob actually fires.** An entry in a loss list that the builder never constructs
  is the #1 failure mode in this repo (pitfall #16): the run smoke-passes while silently
  collapsing to the control's objective. This module builds the real loss and drives a
  gradient through it rather than asserting the YAML says the right thing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from tests.utils.corpus import tracked_yamls
from tests.utils.repo_scripts import require_repo_file

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COHORT_REL = "experiments/inprogress/kspace_filling/sharpness"
_BASELINE_REL = f"{_COHORT_REL}/experiment_11_sharp_baseline.yaml"
_VARIANT_REL = f"{_COHORT_REL}/experiment_11_sharp_hfen.yaml"


def _baseline_path() -> Path:
    return require_repo_file(_BASELINE_REL)


def _variant_path() -> Path:
    return require_repo_file(_VARIANT_REL)

# Selection is held IDENTICAL across the pair, so it is not a second variable.
_SELECTION_METRIC = "val_hfen_mean"


def _entries(settings: TrainingSettings, section: str) -> list[dict[str, Any]]:
    raw = getattr(settings.losses, section, None) or []
    return [e if isinstance(e, dict) else e.model_dump() for e in raw]


@pytest.fixture(scope="module")
def baseline() -> TrainingSettings:
    return TrainingSettings.from_yaml(str(_baseline_path()))


@pytest.fixture(scope="module")
def variant() -> TrainingSettings:
    return TrainingSettings.from_yaml(str(_variant_path()))


def test_both_arms_exist() -> None:
    # require_repo_file, not `.is_file()`: an arm the allowlist never selects is
    # a publication boundary (skip), while an arm that was deleted or MOVED is
    # the ablation breaking, and must stay loud in every tree but the export.
    assert _baseline_path().is_file() and _variant_path().is_file()


def test_hfen_is_the_only_loss_delta(
    baseline: TrainingSettings, variant: TrainingSettings
) -> None:
    """`hfen` in image_losses is the entire objective difference."""
    for section in ("kspace_losses", "complex_losses"):
        assert _entries(baseline, section) == _entries(
            variant, section
        ), f"{section} differs between the arms; the ablation is confounded (#17)."
    assert _entries(baseline, "image_losses") == []
    names = [e["name"] for e in _entries(variant, "image_losses")]
    assert names == [
        "hfen"
    ], f"variant image_losses = {names!r}, expected exactly ['hfen']"


def test_selection_metric_is_identical_and_not_psnr(
    baseline: TrainingSettings, variant: TrainingSettings
) -> None:
    """PSNR rewards blur, so selecting on it would fight the thing being measured.

    In the attention_none run PSNR ranked the two blurriest cases (7.8% of target
    high-band power) at 42.2 / 41.4 dB and the one case matching high-band power (103%)
    at 33.4 dB. ``val_hfen_mean`` is a LoG-filtered error norm against the TRUE target,
    so it cannot reward hallucinated texture. Held identical across the pair.
    """
    for settings, label in ((baseline, "baseline"), (variant, "variant")):
        assert settings.metrics.best_metric_name == _SELECTION_METRIC, label
        assert (
            str(getattr(settings.metrics.best_metric_mode, "value", "")) == "min"
        ), label
        assert settings.early_stopping.metric == _SELECTION_METRIC, label
        assert str(getattr(settings.early_stopping.mode, "value", "")) == "min", label


def test_selection_metric_is_actually_produced(
    baseline: TrainingSettings, variant: TrainingSettings
) -> None:
    """``val_hfen_mean`` only exists if ``hfen`` is in ``validation.metrics`` (#18).

    ``_stamp_accel_mean`` averages the per-level ``val_<m>_<R>x`` columns for the
    CONFIGURED validation metrics. Before 2026-07-29 that set was hardcoded to
    psnr/robust_mri_psnr, so this selection target would never have been written.
    """
    for settings, label in ((baseline, "baseline"), (variant, "variant")):
        declared = [str(getattr(m, "value", m)) for m in settings.validation.scoring.compute]
        base = _SELECTION_METRIC.removeprefix("val_").removesuffix("_mean")
        assert base in declared, (
            f"{label}: selects on {_SELECTION_METRIC} but validation.metrics={declared} "
            f"never emits {base} -- selection on a key the run does not compute (#18)."
        )


def test_hfen_loss_fires_and_is_differentiable() -> None:
    """Build the REAL registered loss and drive a gradient through it.

    A loss entry the builder never constructs is pitfall #16: the arm smoke-passes while
    training the control's objective. Asserting on YAML cannot catch that, so this
    resolves the registry, constructs with the arm's own declared kwargs, and checks the
    term both discriminates a blurred input and produces a non-zero gradient.
    """
    from spectramr.models.losses.registry import LossRegistry

    settings = TrainingSettings.from_yaml(str(_variant_path()))
    entry = _entries(settings, "image_losses")[0]
    loss_fn = LossRegistry.create(entry["name"], **entry["kwargs"])

    torch.manual_seed(0)
    target = torch.rand(1, 1, 64, 64)
    # A blurred prediction must score WORSE than the target itself, or the term is not
    # measuring high-frequency content at all.
    kernel = torch.ones(1, 1, 5, 5) / 25.0
    blurred = torch.nn.functional.conv2d(target, kernel, padding=2)

    perfect = float(loss_fn(target.clone(), target))
    smoothed = float(loss_fn(blurred, target))
    assert smoothed > perfect, (
        f"hfen did not penalise blur (blurred={smoothed:.6f} <= exact={perfect:.6f}); "
        "the term is inert as configured."
    )

    pred = blurred.clone().requires_grad_(True)
    loss = loss_fn(pred, target) * entry["weight"]
    loss.backward()
    assert pred.grad is not None and torch.any(
        pred.grad != 0
    ), "hfen produced no gradient — it would contribute nothing to training."


def test_hfen_does_not_double_bridge() -> None:
    """``image_losses`` are iFFT-bridged once by the builder under output_domain: kspace.

    A loss that ALSO bridges internally (``use_fourier_bridge=True``) gets transformed
    twice — the #467 failure, which silently redirected 47% of that cohort's loss weight
    into the wrong domain. HFEN must not carry the flag.
    """
    from spectramr.models.losses.registry import LossRegistry

    settings = TrainingSettings.from_yaml(str(_variant_path()))
    assert str(
        getattr(settings.losses.policy.output_domain, "value", settings.losses.policy.output_domain)
    ) == ("kspace")
    entry = _entries(settings, "image_losses")[0]
    loss_fn = LossRegistry.create(entry["name"], **entry["kwargs"])
    assert not getattr(loss_fn, "use_fourier_bridge", False), (
        "hfen declares use_fourier_bridge; combined with the image_losses bridge it "
        "would be iFFT'd twice (#467)."
    )


def test_declared_kwargs_survive_the_schema() -> None:
    """Per-entry ``kwargs`` must reach the resolved object.

    ``LossComponentConfig`` is ``extra="ignore"``, so the sibling key ``config:`` is
    silently dropped — which is exactly how ``sobolev_order`` became a dead knob across
    53 arms (issue #615). This pins the surface that actually works.
    """
    settings = TrainingSettings.from_yaml(str(_variant_path()))
    kwargs = _entries(settings, "image_losses")[0]["kwargs"]
    assert kwargs == {
        "kernel_size": 15,
        "sigma": 1.5,
        "normalize": True,
    }, f"hfen kwargs resolved to {kwargs!r} — the declared knobs did not survive load."


# ---------------------------------------------------------------------------
# The hard-DC arms (2026-07-29): attention_shootout carries the same term.
#
# Which arms CAN show sharp images is decided by data consistency, not by the
# loss. ``SimpleDataConsistency`` in k-space returns, at sampled lines:
#
#     hard:  k = measured                              (100% of the measurement)
#     soft:  k = (pred + w*measured) / (1 + w)         (w=0.5 -> 33%)
#
# So a soft-DC arm emits 2/3 prediction + 1/3 measurement at the very lines
# whose truth it already HAS. hard is now the cohort standard (48 of 58 arms);
# the 9 exemptions are the arms where DC is the INDEPENDENT VARIABLE -- all of
# dc_shootout/, the two CNN-ADC ablations, and the two cross-contrast arms that
# declare a named non-default method (#611). The loss term below is a separate
# lever, applied to attention_shootout only.
# ---------------------------------------------------------------------------

_SHOOTOUT = _REPO_ROOT / "experiments" / "inprogress" / "kspace_filling" / (
    "attention_shootout"
)


def _shootout_arms() -> list[Path]:
    return tracked_yamls(_SHOOTOUT, "experiment_11_attention_*.yaml", recursive=False)


def test_shootout_is_the_hard_dc_family() -> None:
    """The premise for applying the term here: the measurement survives DC."""
    for path in _shootout_arms():
        settings = TrainingSettings.from_yaml(str(path))
        method = settings.physics.data_consistency.method
        assert str(getattr(method, "value", method)) == "hard", (
            f"{path.name} is not hard-DC ({method!r}); its measured k-space is "
            "attenuated before it reaches the output, so a high-frequency loss "
            "term cannot be the binding constraint."
        )


@pytest.mark.parametrize("path", _shootout_arms(), ids=lambda p: p.stem)
def test_shootout_arm_carries_the_hf_term_uniformly(path: Path) -> None:
    """Uniform across the family, so attention type stays the only variable."""
    settings = TrainingSettings.from_yaml(str(path))
    entries = _entries(settings, "image_losses")
    assert [e["name"] for e in entries] == ["hfen"], (
        f"{path.name} image_losses = {[e['name'] for e in entries]!r}; a term "
        "present on some arms and absent on others confounds the shootout (#17)."
    )
    assert entries[0]["weight"] == 0.3
    assert entries[0]["kwargs"] == {
        "kernel_size": 15,
        "sigma": 1.5,
        "normalize": True,
    }


@pytest.mark.parametrize("path", _shootout_arms(), ids=lambda p: p.stem)
def test_shootout_arm_does_not_select_on_psnr(path: Path) -> None:
    """PSNR provably ranks the blurriest checkpoint first on this arm family.

    Measured on the attention_none run: the two cases carrying 7.8% of the
    target's high-band power scored 42.2 / 41.4 dB, and the one case matching it
    (103%) scored 33.4 dB.
    """
    settings = TrainingSettings.from_yaml(str(path))
    assert settings.metrics.best_metric_name == _SELECTION_METRIC, path.name
    assert str(getattr(settings.metrics.best_metric_mode, "value", "")) == "min"
    assert settings.early_stopping.metric == _SELECTION_METRIC, path.name
    assert str(getattr(settings.early_stopping.mode, "value", "")) == "min"

    declared = [str(getattr(m, "value", m)) for m in settings.validation.scoring.compute]
    assert "hfen" in declared, (
        f"{path.name} selects on {_SELECTION_METRIC} but validation.metrics="
        f"{declared} never emits hfen — selection on a key the run does not "
        "compute (#18)."
    )
