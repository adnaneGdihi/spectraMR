"""D01#4 (callee half): ablation must be able to represent "no device requested".

``spectramr ablation`` parses ``--device`` with ``default=None``, but the CLI then
did ``or "cuda"`` before calling in, because every function on the ablation path
was typed ``device: str = "cuda"``. Removing the CLI fallback without widening
the callees would have replaced a hardcoded device with a lie in the signature.

``None`` is not "CPU" -- it means *the config, then the 9b resolver, decides*,
which is exactly what ``run_training_pipeline(device: str | None = None)`` at
the end of the chain already accepted.
"""

import inspect

import pytest

from spectramr.pipelines import ablation as ablation_mod
from spectramr.pipelines.train import run_training_pipeline


@pytest.mark.parametrize(
    "func",
    [
        ablation_mod.train_and_score,
        ablation_mod.run_ablation_study,
        ablation_mod.run_loss_ablation,
    ],
    ids=lambda f: f.__name__,
)
def test_device_parameter_is_optional(func):
    param = inspect.signature(func).parameters["device"]
    assert param.default is None, (
        f"{func.__name__} defaults device to {param.default!r}; a hardcoded "
        "device here re-imposes what the CLI stopped imposing."
    )
    assert param.annotation in ("str | None", str | None)


def test_the_terminal_consumer_already_accepted_none():
    """Pins the reason ``None`` is safe to forward: the chain ends here."""
    assert inspect.signature(run_training_pipeline).parameters["device"].default is None


def test_train_and_score_forwards_none_unchanged(monkeypatch):
    seen = {}

    def _fake_run_training_pipeline(config, device=None, **kwargs):
        seen["device"] = device
        return {"success": False, "error": "stopped before training"}

    import spectramr.pipelines.train as train_mod

    monkeypatch.setattr(train_mod, "run_training_pipeline", _fake_run_training_pipeline)

    ablation_mod.train_and_score(config=object())
    assert seen["device"] is None, "an unrequested device must not become 'cuda'"


def test_train_and_score_forwards_an_explicit_device(monkeypatch):
    seen = {}

    def _fake_run_training_pipeline(config, device=None, **kwargs):
        seen["device"] = device
        return {"success": False, "error": "stopped before training"}

    import spectramr.pipelines.train as train_mod

    monkeypatch.setattr(train_mod, "run_training_pipeline", _fake_run_training_pipeline)

    ablation_mod.train_and_score(config=object(), device="cpu")
    assert seen["device"] == "cpu"


def test_a_result_without_a_success_flag_is_not_scored(monkeypatch):
    """#181: an absent flag used to default to True and the variant was scored."""
    import spectramr.pipelines.train as train_mod

    monkeypatch.setattr(train_mod, "run_training_pipeline", lambda *a, **k: {})
    result = ablation_mod.train_and_score(config=object())
    assert result["success"] is False
    assert "no success flag" in result["error"]


@pytest.mark.parametrize(
    ("final_loss", "expected"),
    [(None, "absent"), (0.25, 0.25)],
    ids=["none_is_omitted", "float_passes_through"],
)
def test_final_loss_none_does_not_raise(monkeypatch, tmp_path, final_loss, expected):
    """The training result carries ``final_loss=None`` when no g_total_loss was recorded."""
    import spectramr.pipelines.train as train_mod

    monkeypatch.setattr(
        train_mod,
        "run_training_pipeline",
        lambda *a, **k: {"success": True, "final_loss": final_loss},
    )
    monkeypatch.setattr(
        ablation_mod, "_resolve_validation_metrics_csv", lambda cfg: tmp_path / "none.csv"
    )
    monkeypatch.setattr(train_mod, "_summarise_best_metrics_from_csv", lambda p: {})

    metrics = ablation_mod.train_and_score(config=object())

    assert metrics["success"] is True
    if expected == "absent":
        assert "final_loss" not in metrics
    else:
        assert metrics["final_loss"] == expected
        assert isinstance(metrics["final_loss"], float)
