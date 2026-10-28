"""Tests for the scripting entry point :func:`spectramr.pipelines.fit.fit`.

``fit`` assembles a :class:`TrainingEnvironment` from hand-supplied objects and
drives the SAME ``run_training_pipeline`` the config path uses. These tests
verify the *assembly + injection* wiring (with a real ``LossBuilder`` so the
losses are genuinely built — not a mocked facade), while stubbing the pipeline
call itself so no full training loop runs here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import spectramr.pipelines.fit as fit_mod
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.pipelines.fit import Trainer, fit


def _loader() -> DataLoader:
    return DataLoader(TensorDataset(torch.randn(2, 2, 8, 8)), batch_size=1)


def test_fit_assembles_env_and_injects_into_pipeline(monkeypatch):
    captured: dict = {}

    def fake_run(settings, *, device=None, env=None, strategy=None):
        captured.update(settings=settings, device=device, env=env, strategy=strategy)
        return {"success": True}

    monkeypatch.setattr(fit_mod, "run_training_pipeline", fake_run)

    model = nn.Conv2d(2, 2, 3, padding=1)
    result = fit(model, _loader(), device="cpu")

    assert result == {"success": True}
    env = captured["env"]
    assert isinstance(env, TrainingEnvironment)
    assert env.generator is model  # placed under the canonical key the loop reads
    assert env.opt_g is not None  # optimizer built from config
    assert env.losses  # NON-empty: real LossBuilder output, genuinely consumed
    # Default path lets the pipeline resolve the reconstruction strategy.
    assert captured["strategy"] is None
    assert captured["settings"].training.strategy_class == "reconstruction"


def test_fit_uses_provided_optimizer_and_losses(monkeypatch):
    captured: dict = {}

    def fake_run(settings, **kwargs):
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(fit_mod, "run_training_pipeline", fake_run)

    model = nn.Conv2d(2, 2, 3, padding=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = {"l1": nn.L1Loss()}

    fit(model, _loader(), optimizer=opt, losses=losses, device="cpu")

    env = captured["env"]
    assert env.opt_g is opt  # user optimizer used verbatim
    assert env.losses is losses  # user losses used verbatim (no LossBuilder)


def test_fit_passes_custom_strategy_through(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        fit_mod,
        "run_training_pipeline",
        lambda s, **k: captured.update(k) or {"success": True},
    )
    sentinel_strategy = object()
    fit(
        nn.Conv2d(2, 2, 3, padding=1),
        _loader(),
        losses={"l1": nn.L1Loss()},
        strategy=sentinel_strategy,
        device="cpu",
    )
    assert captured["strategy"] is sentinel_strategy


def test_trainer_applies_stored_epochs(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        fit_mod,
        "run_training_pipeline",
        lambda s, **k: captured.update(settings=s) or {"success": True},
    )
    Trainer(device="cpu", epochs=3).fit(
        nn.Conv2d(2, 2, 3, padding=1), _loader(), losses={"l1": nn.L1Loss()}
    )
    assert captured["settings"].training.epochs == 3


# ---------------------------------------------------------------------------
# Multi-paradigm support — fit() wires each paradigm's env/config non-facade.
# Each test stubs run_training_pipeline and inspects the assembled config/env;
# the strategies' actual loss-consumption is covered by their own integration
# tests. The GAN env-key checks are the real non-facade teeth (without a
# discriminator + opt_d, GAN silently collapses to plain L1).
# ---------------------------------------------------------------------------


def _capture(monkeypatch) -> dict:
    captured: dict = {}

    def fake_run(settings, *, device=None, env=None, strategy=None):
        captured.update(settings=settings, env=env, strategy=strategy)
        return {"success": True}

    monkeypatch.setattr(fit_mod, "run_training_pipeline", fake_run)
    return captured


def test_fit_rejects_unknown_paradigm():
    with pytest.raises(ValueError, match="paradigm"):
        fit(nn.Conv2d(2, 2, 3, padding=1), _loader(), paradigm="not_a_real_paradigm")


def test_fit_reconstruction_default_paradigm(monkeypatch):
    cap = _capture(monkeypatch)
    fit(nn.Conv2d(2, 2, 3, padding=1), _loader(), device="cpu")
    assert cap["settings"].training.strategy_class == "reconstruction"
    assert cap["env"].discriminator is None  # single-model paradigm


def test_fit_diffusion_sets_diffusion_config(monkeypatch):
    cap = _capture(monkeypatch)
    fit(nn.Conv2d(2, 2, 3, padding=1), _loader(), paradigm="diffusion", device="cpu")
    cfg = cap["settings"]
    assert cfg.training.strategy_class == "diffusion"
    assert cfg.training.diffusion is not None
    assert cfg.training.diffusion.timesteps == 1000
    assert cap["env"].discriminator is None


def test_fit_vae_sets_latent_loss_config(monkeypatch):
    cap = _capture(monkeypatch)
    fit(nn.Conv2d(2, 2, 3, padding=1), _loader(), paradigm="vae", device="cpu")
    cfg = cap["settings"]
    assert cfg.training.strategy_class == "vae"
    # the kl/latent loss config is present -> LossBuilder builds the latent loss
    assert cfg.losses.latent is not None


def test_fit_gan_wires_discriminator_and_opt_d(monkeypatch):
    cap = _capture(monkeypatch)
    generator = nn.Conv2d(2, 2, 3, padding=1)
    discriminator = nn.Conv2d(2, 1, 3, padding=1)
    fit(
        generator,
        _loader(),
        paradigm="gan",
        discriminator=discriminator,
        device="cpu",
    )
    env = cap["env"]
    assert env.generator is generator
    assert env.discriminator is discriminator  # GAN non-facade: D is wired
    assert env.opt_d is not None  # and its optimizer
    settings = cap["settings"]
    assert settings.training.strategy_class == "gan"
    # The ADVERSARIAL TERM must be enabled — not just the discriminator wired.
    # enable_adversarial gates it; without it (the B2 facade) get_enabled_losses()
    # returns only {"l1"} and the GAN trains a vanilla L1 denoiser while the
    # discriminator is built but never updated (pitfall #16).
    assert settings.losses.gan.enable_adversarial is True
    assert "adversarial" in settings.losses.get_enabled_losses()


def test_fit_gan_without_discriminator_raises():
    # No discriminator= and no config disc block -> GAN would collapse to L1;
    # fit() must raise rather than silently degrade (pitfall #9).
    with pytest.raises((ValueError, RuntimeError), match="iscriminator"):
        fit(nn.Conv2d(2, 2, 3, padding=1), _loader(), paradigm="gan", device="cpu")


def test_fit_gan_honours_the_two_timescale_update_rule(monkeypatch):
    """D's optimizer is built here, so TTUR has to be requested here.

    ``optimization.optimizer.discriminator_learning_rate`` is read by the config-driven
    builder via ``role="discriminator"``. The leaf builder had no role seam, so
    ``fit(paradigm='gan')`` built D at G's LR -- TTUR silently off, and the same
    YAML trained a different objective under ``spectramr train``.
    """
    cap = _capture(monkeypatch)
    fit(
        nn.Conv2d(2, 2, 3, padding=1),
        _loader(),
        paradigm="gan",
        discriminator=nn.Conv2d(2, 1, 3, padding=1),
        device="cpu",
        config={
            "optimization": {
                "learning_rate": 1e-4,
                "discriminator_learning_rate": 2.5e-5,
            }
        },
    )
    env = cap["env"]
    assert env.opt_g.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert env.opt_d.param_groups[0]["lr"] == pytest.approx(2.5e-5)


@pytest.mark.parametrize(
    "variant",
    [
        "flow_matching",
        "rectified_flow",
        "resetting_diffusion",
        "tissue_diffusion_pretrain",
        # distinct subclasses with the identical training.diffusion contract
        "flow_matching_pfode",
        "stochastic_interpolants",
        "x_diffusion",
        "cross_modal_diffusion",
    ],
)
def test_fit_diffusion_family_aliases_inherit_diffusion_defaults(monkeypatch, variant):
    """Variants routing to the exact DiffusionTrainingStrategy share the
    `diffusion` default (training.diffusion) — same strategy + config contract,
    so they're provably non-facade aliases of the diffusion paradigm."""
    cap = _capture(monkeypatch)
    fit(nn.Conv2d(2, 2, 3, padding=1), _loader(), paradigm=variant, device="cpu")
    cfg = cap["settings"]
    assert cfg.training.strategy_class == variant  # resolves to its own strategy
    assert cfg.training.diffusion is not None  # inherited the diffusion default
    assert cfg.training.diffusion.timesteps == 1000


def test_fit_recon_routed_paradigm_uses_recon_defaults(monkeypatch):
    """A paradigm whose strategy IS ReconstructionTrainingStrategy (e.g.
    'dual_task') is a recon variant — generic recon defaults are correct, so
    fit(paradigm=...) works with no extra config."""
    cap = _capture(monkeypatch)
    fit(nn.Conv2d(2, 2, 3, padding=1), _loader(), paradigm="dual_task", device="cpu")
    assert cap["settings"].training.strategy_class == "dual_task"


def test_fit_fisher_rao_flow_takes_recon_defaults_without_diffusion_block(monkeypatch):
    """fisher_rao_flow routes to ReconstructionTrainingStrategy, which refuses training.diffusion."""
    cap = _capture(monkeypatch)
    fit(nn.Conv2d(2, 2, 3, padding=1), _loader(), paradigm="fisher_rao_flow", device="cpu")
    assert "diffusion" not in cap["settings"].training.model_fields_set


def test_fit_nondefault_paradigm_without_config_raises():
    """A valid-but-untailored, non-recon paradigm has no faithful synthetic
    default — fit() must RAISE with guidance rather than silently mis-default
    (pitfall #9), unless a config / strategy is supplied."""
    with pytest.raises(ValueError, match=r"no tailored default|facade"):
        fit(nn.Conv2d(2, 2, 3, padding=1), _loader(), paradigm="pnp", device="cpu")


def test_fit_nondefault_paradigm_with_config_is_trusted(monkeypatch):
    """When the caller explicitly configures an untailored paradigm, fit trusts
    the config (no raise)."""
    cap = _capture(monkeypatch)
    fit(
        nn.Conv2d(2, 2, 3, padding=1),
        _loader(),
        paradigm="pnp",
        config={"training": {"strategy_class": "pnp", "epochs": 2}},
        device="cpu",
    )
    assert cap["settings"].training.strategy_class == "pnp"
    assert cap["settings"].training.epochs == 2


@pytest.mark.parametrize(
    "paradigm,distinctive",
    [
        ("edm", lambda c: c.training.diffusion.parameterization == "edm"),
        ("masked", lambda c: c.training.masked is not None),
        ("cartoon_texture_safe", lambda c: c.training.cartoon_texture_safe is not None),
        ("recoverability_vib", lambda c: c.training.recoverability_vib is not None),
        ("generative", lambda c: True),  # generic default; strategy reads only base
        ("universal_reconstruction", lambda c: True),  # alias -> reconstruction
    ],
)
def test_fit_bespoke_tailored_defaults(monkeypatch, paradigm, distinctive):
    """Verified-non-facade bespoke defaults: fit assembles the paradigm's own
    strategy_class + the distinctive config block its strategy actually reads."""
    cap = _capture(monkeypatch)
    fit(nn.Conv2d(2, 2, 3, padding=1), _loader(), paradigm=paradigm, device="cpu")
    cfg = cap["settings"]
    assert cfg.training.strategy_class == paradigm
    assert distinctive(cfg)


def test_fit_val_loader_defaults_to_train_when_unset(monkeypatch):
    """val_loader unset (None) → the train loader is reused for validation."""
    cap = _capture(monkeypatch)
    train = _loader()
    fit(nn.Conv2d(2, 2, 3, padding=1), train, val_loader=None, device="cpu")
    dl = cap["env"].data_loaders
    assert dl["train"] is train
    assert dl["val"] is train  # fell back to the train loader


def test_fit_uses_provided_val_loader_even_when_empty(monkeypatch):
    """A real val loader is used verbatim — including an EMPTY one. An empty
    DataLoader is falsy (``len()==0``), so the old ``val_loader or train_loader``
    would have silently swapped in the train loader; ``is not None`` keeps it."""
    cap = _capture(monkeypatch)
    empty_val = DataLoader(TensorDataset(torch.empty(0, 2, 8, 8)), batch_size=1)
    assert not empty_val  # empty loader is falsy — the footgun
    train = _loader()
    fit(nn.Conv2d(2, 2, 3, padding=1), train, val_loader=empty_val, device="cpu")
    dl = cap["env"].data_loaders
    assert dl["val"] is empty_val
    assert dl["val"] is not dl["train"]


def test_fit_jepa_is_config_only(monkeypatch):
    """jepa needs an encoder->embeddings model; a generic default would facade
    (pitfall #16), so it must remain config-only and raise without a config."""
    with pytest.raises(ValueError, match=r"no tailored default|facade"):
        fit(nn.Conv2d(2, 2, 3, padding=1), _loader(), paradigm="jepa", device="cpu")


# ---------------------------------------------------------------------------
# L3: _build_discriminator_from_config must not swallow a REAL build failure as
# "no discriminator" (which mis-routes the user to the "pass discriminator=" fix).
# ---------------------------------------------------------------------------


class _BoomBuilder:
    def __init__(self, *a, **k):
        pass

    def build_discriminator(self):
        return self

    def build(self):
        raise RuntimeError("kaboom")


def test_build_discriminator_reraises_when_configured_but_build_fails(monkeypatch):
    """A configured discriminator that fails to build → RuntimeError (not None).

    The function only reads ``settings.model.discriminator_component`` and hands
    ``settings`` to the (mocked) ModelBuilder, so a SimpleNamespace mock suffices
    — no need to round-trip a full pydantic config."""
    from types import SimpleNamespace

    import spectramr.infrastructure.training.builders.model_builder as mb_mod

    monkeypatch.setattr(mb_mod, "ModelBuilder", _BoomBuilder)
    settings = SimpleNamespace(
        model=SimpleNamespace(discriminator_component="patch_gan")
    )
    with pytest.raises(RuntimeError, match="configured but failed"):
        fit_mod._build_discriminator_from_config(settings, torch.device("cpu"))


def test_build_discriminator_returns_none_when_unconfigured(monkeypatch):
    """No discriminator_component → genuinely absent → None (not a raise)."""
    from types import SimpleNamespace

    import spectramr.infrastructure.training.builders.model_builder as mb_mod

    monkeypatch.setattr(mb_mod, "ModelBuilder", _BoomBuilder)
    settings = SimpleNamespace(model=SimpleNamespace(discriminator_component=None))
    assert (
        fit_mod._build_discriminator_from_config(settings, torch.device("cpu")) is None
    )


# ---------------------------------------------------------------------------
# WS-3 PR-4: Trainer.evaluate() (in-memory) + Trainer.predict() (path-based)
# ---------------------------------------------------------------------------


def test_trainer_evaluate_builds_env_and_runs_validation(monkeypatch):
    """``Trainer.evaluate`` assembles the same env ``fit`` would (caller's loader
    under the ``"val"`` key the validator reads) and returns the metrics from the
    shared ``TrainingLoop.evaluate``. The strategy build + validation pass are
    stubbed so this stays a light wiring test (no real loop = no dev-box OOM)."""
    captured: dict = {}

    class _FakeLoop:
        def evaluate(self):
            return {"val_psnr": 28.0}

    def fake_build(settings, env, *, device, strategy=None):
        captured.update(settings=settings, env=env, device=device, strategy=strategy)
        return _FakeLoop()

    monkeypatch.setattr(fit_mod, "_build_evaluation_loop", fake_build)

    model = nn.Conv2d(2, 2, 3, padding=1)
    val = _loader()
    result = Trainer(device="cpu").evaluate(model, val)

    assert result == {"val_psnr": 28.0}
    env = captured["env"]
    assert isinstance(env, TrainingEnvironment)
    assert env.generator is model  # canonical generator key
    assert env.data_loaders["val"] is val  # caller's loader is what gets validated
    assert env.opt_g is not None  # env requires an optimizer (never stepped)
    assert captured["strategy"] is None  # factory resolves it by default


def test_trainer_evaluate_passes_custom_strategy_through(monkeypatch):
    """A ``Trainer(strategy=...)`` must skip the factory in evaluate too."""
    sentinel = object()
    captured: dict = {}

    class _FakeLoop:
        def evaluate(self):
            return {}

    def fake_build(settings, env, *, device, strategy=None):
        captured["strategy"] = strategy
        return _FakeLoop()

    monkeypatch.setattr(fit_mod, "_build_evaluation_loop", fake_build)

    Trainer(device="cpu", strategy=sentinel).evaluate(
        nn.Conv2d(2, 2, 3, padding=1), _loader()
    )
    assert captured["strategy"] is sentinel


def test_trainer_predict_delegates_to_inference_pipeline(monkeypatch):
    """``Trainer.predict`` is a thin, path-based delegate to the SSOT
    ``run_inference_pipeline`` (config + checkpoint driven; writes via the data
    layer). It wraps the four path args in ``Path`` and forwards device/batch."""
    from pathlib import Path

    import spectramr.pipelines.infer as infer_mod

    captured: dict = {}

    def fake_infer(
        config_path,
        checkpoint_path,
        input_path,
        output_path,
        *,
        device="cuda",
        batch_size=None,
    ):
        captured.update(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            input_path=input_path,
            output_path=output_path,
            device=device,
            batch_size=batch_size,
        )
        return {"outputs": ["recon_0.nii"]}

    monkeypatch.setattr(infer_mod, "run_inference_pipeline", fake_infer)

    result = Trainer(device="cpu").predict(
        config_path="cfg.yaml",
        checkpoint_path="best.pt",
        input_path="in_dir/",
        output_path="out_dir/",
        batch_size=4,
    )

    assert result == {"outputs": ["recon_0.nii"]}
    assert captured["config_path"] == Path("cfg.yaml")
    assert captured["checkpoint_path"] == Path("best.pt")
    assert captured["input_path"] == Path("in_dir/")
    assert captured["output_path"] == Path("out_dir/")
    assert captured["device"] == "cpu"
    assert captured["batch_size"] == 4


# ---------------------------------------------------------------------------
# fit() must not report failure only in a return value.
#
# ``run_training_pipeline`` returns ``{"success": False, "error": ...}`` because
# the CLI turns that into an exit code. A script has no exit code to read, so a
# caller who forgets ``if result["success"]`` carries on with an UNTRAINED model
# that looks entirely normal.
# ---------------------------------------------------------------------------


def test_fit_raises_by_default_when_the_run_fails(monkeypatch):
    """The regression guard: a failed run must not return quietly."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    import spectramr.pipelines.fit as fit_mod

    class _DS(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {"input": torch.randn(1, 8, 8), "target": torch.randn(1, 8, 8)}

    monkeypatch.setattr(
        fit_mod,
        "run_training_pipeline",
        lambda *a, **k: {"success": False, "error": "boom"},
    )
    with pytest.raises(fit_mod.TrainingFailedError, match="boom"):
        fit_mod.fit(nn.Conv2d(1, 1, 3, padding=1), DataLoader(_DS(), batch_size=1), device="cpu")


def test_fit_returns_the_dict_when_opted_out(monkeypatch):
    """``raise_on_failure=False`` preserves the inspect-it-yourself path."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    import spectramr.pipelines.fit as fit_mod

    class _DS(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {"input": torch.randn(1, 8, 8), "target": torch.randn(1, 8, 8)}

    monkeypatch.setattr(
        fit_mod,
        "run_training_pipeline",
        lambda *a, **k: {"success": False, "error": "boom"},
    )
    result = fit_mod.fit(
        nn.Conv2d(1, 1, 3, padding=1),
        DataLoader(_DS(), batch_size=1),
        device="cpu",
        raise_on_failure=False,
    )
    assert result["success"] is False and result["error"] == "boom"


def test_fit_does_not_raise_on_success(monkeypatch):
    """A successful run is untouched -- the guard must not fire on the happy path."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    import spectramr.pipelines.fit as fit_mod

    class _DS(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {"input": torch.randn(1, 8, 8), "target": torch.randn(1, 8, 8)}

    monkeypatch.setattr(
        fit_mod, "run_training_pipeline", lambda *a, **k: {"success": True, "best_metrics": {}}
    )
    result = fit_mod.fit(
        nn.Conv2d(1, 1, 3, padding=1), DataLoader(_DS(), batch_size=1), device="cpu"
    )
    assert result["success"] is True


def test_trainer_threads_raise_on_failure():
    """``Trainer`` must carry the option, or the fluent form silently opts out."""
    import inspect

    from spectramr.pipelines.fit import Trainer

    assert "raise_on_failure" in inspect.signature(Trainer.__init__).parameters
