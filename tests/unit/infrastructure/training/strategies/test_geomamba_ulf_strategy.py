"""Unit tests for GeoMambaULFStrategy wiring (2026-06 Tier-1 fixes).

Covers two inert-mechanism (#16/#15) repairs:

* The metric-SFC SSOT block (``config.training.geomamba_ulf.metric_sfc``) is
  bridged onto the generator. The generator is built from ``config.model`` and
  never sees the ``geomamba_ulf`` training block, so without the bridge the
  whole ``MetricSFCConfig`` (beta / smoothing_sigma / connectivity / cache_dir)
  was inert — ``abl_no_metric_sfc`` (beta=0) was byte-identical to ``v0`` and
  the beta-HPO sweep was six identical runs.
* ``_extract_contrast_id`` reads ``contrast_idx`` in addition to
  ``contrast_id``. The loaders (m4raw / slice_dataset) and the default
  collation emit ``contrast_idx``; reading only ``contrast_id`` silently
  dropped it, collapsing FiLM conditioning to the same (gamma, beta) for every
  contrast.

The full ``__init__`` builds a DI container + AMP + an optimizer, so the
device-agnostic helpers are exercised on a bare ``__new__`` instance.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")
pytest.importorskip("scipy")

from spectramr.infrastructure.training.strategies.geomamba_ulf_strategy import (  # noqa: E402
    GeoMambaULFStrategy,
)
from spectramr.models.generators.geo_mamba_unet import GeoMambaUNet  # noqa: E402


def _tiny_model() -> GeoMambaUNet:
    return GeoMambaUNet(in_channels=1, out_channels=1, d_model=8, n_blocks=1)


# ---------------------------------------------------------------------------
# 1.4 — contrast_idx is read (FiLM conditioning fires)
# ---------------------------------------------------------------------------


def test_extract_contrast_id_reads_contrast_idx() -> None:
    """The standard ``contrast_idx`` key (m4raw / slice_dataset / collation)
    must be honoured, not silently dropped."""
    cid = GeoMambaULFStrategy._extract_contrast_id(
        {"contrast_idx": torch.tensor([2, 1])}, torch.zeros(2, 1, 4, 4)
    )
    assert cid is not None
    assert cid.tolist() == [2, 1]
    assert cid.dtype == torch.long


def test_extract_contrast_id_prefers_contrast_id_over_idx() -> None:
    """When both keys are present ``contrast_id`` wins (mrixfields convention)."""
    cid = GeoMambaULFStrategy._extract_contrast_id(
        {"contrast_id": torch.tensor([0]), "contrast_idx": torch.tensor([3])},
        torch.zeros(1, 1, 4, 4),
    )
    assert cid.tolist() == [0]


def test_extract_contrast_id_none_when_absent() -> None:
    cid = GeoMambaULFStrategy._extract_contrast_id({}, torch.zeros(1))
    assert cid is None


# ---------------------------------------------------------------------------
# 1.5 — metric_sfc SSOT is bridged onto the generator
# ---------------------------------------------------------------------------


def test_apply_metric_sfc_config_bridges_onto_model() -> None:
    """``geo_cfg.metric_sfc`` values land on the generator before any forward."""
    model = _tiny_model()
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=model)
    geo_cfg = SimpleNamespace(
        metric_sfc=SimpleNamespace(
            beta=0.0, smoothing_sigma=3.0, connectivity=18, cache_dir="cache/x"
        )
    )
    s._apply_metric_sfc_config(geo_cfg)
    assert model.sfc_beta == 0.0
    assert model.sfc_smoothing_sigma == 3.0
    assert model.sfc_connectivity == 18
    assert model.sfc_cache_dir == "cache/x"


def test_apply_metric_sfc_config_noop_without_block() -> None:
    """No ``metric_sfc`` block → the model keeps its construction defaults."""
    model = _tiny_model()
    before = (model.sfc_beta, model.sfc_smoothing_sigma, model.sfc_connectivity)
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=model)
    s._apply_metric_sfc_config(SimpleNamespace())  # geo_cfg has no metric_sfc
    s._apply_metric_sfc_config(None)  # geo_cfg absent entirely
    assert (model.sfc_beta, model.sfc_smoothing_sigma, model.sfc_connectivity) == before


def test_apply_metric_sfc_config_unwraps_ddp_module() -> None:
    """A DDP/EMA-wrapped generator (``.module``) is unwrapped before patching."""
    model = _tiny_model()
    wrapper = SimpleNamespace(module=model)
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=wrapper)
    s._apply_metric_sfc_config(SimpleNamespace(metric_sfc=SimpleNamespace(beta=7.0)))
    assert model.sfc_beta == 7.0


# ---------------------------------------------------------------------------
# Commit 1 — PH loss is fail-loud when the [topology] extra is missing
# ---------------------------------------------------------------------------


def test_get_ph_loss_raises_when_topology_deps_missing(monkeypatch) -> None:
    """A non-zero cubical_ph_w2 weight with gudhi/POT absent must RAISE, not
    warn-and-skip. Skipping silently degrades the PH arm to plain L1 while it
    smoke-PASSes (pitfalls #9/#16). The strategy's whole method claim is PH."""
    import spectramr.models.losses.cubical_ph_w2_loss as ph_mod

    class _NoDeps:
        def __init__(self, *a, **k):
            raise ImportError("gudhi/POT not installed")

    monkeypatch.setattr(ph_mod, "CubicalPHWassersteinLoss", _NoDeps)

    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s._ph_loss = None
    s._topology_cfg = None
    with pytest.raises(RuntimeError, match="topology"):
        s._get_ph_loss()


def test_get_ph_loss_caches_constructed_module(monkeypatch) -> None:
    """When deps are present the module is built once and cached."""
    import spectramr.models.losses.cubical_ph_w2_loss as ph_mod

    calls = {"n": 0}

    class _FakePH:
        def __init__(self, *a, **k):
            calls["n"] += 1
            self.homology_dims = tuple(k.get("homology_dims", (0, 1)))
            self.wasserstein_p = k.get("wasserstein_p", 2)
            self.stratify_subvolumes = k.get("stratify_subvolumes", 1)

    monkeypatch.setattr(ph_mod, "CubicalPHWassersteinLoss", _FakePH)

    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s._ph_loss = None
    s._topology_cfg = None
    first = s._get_ph_loss()
    second = s._get_ph_loss()
    assert first is second
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Cohort review 2026-09-03 — one weight owner, a truthful Beltrami, declared
# loss ownership
# ---------------------------------------------------------------------------


def _weights_cfg(entries, reconstruction=None):
    """A config double: ``image_losses`` entries and an optional lambda section."""
    return SimpleNamespace(
        losses=SimpleNamespace(
            image_losses=[{"name": n, "weight": w} for n, w in entries],
            kspace_losses=[],
            complex_losses=[],
            reconstruction=reconstruction if reconstruction is not None else SimpleNamespace(),
        ),
        training=SimpleNamespace(geomamba_ulf=None),
    )


def _bare_for_setup(cfg):
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.config = cfg
    s.env = SimpleNamespace(generator=_tiny_model())
    s._apply_metric_sfc_config = lambda *_a, **_k: None
    s._ulf_unwarp = None
    s.logging_service = None
    return s


def test_the_three_weights_come_from_the_loss_weight_table() -> None:
    """``losses.image_losses`` entries are the one owner: the strategy used to read l1
    from ``losses.reconstruction.lambda_l1`` (schema default 10.0) and the topology
    terms through its own resolver."""
    s = _bare_for_setup(_weights_cfg([("l1", 10.0), ("cubical_ph_w2", 0.1)]))
    GeoMambaULFStrategy._setup_strategy_specific_components(s)
    assert (s._lambda_l1, s._lambda_ph_w2, s._lambda_beltrami) == (10.0, 0.1, 0.0)


def test_an_author_written_lambda_is_the_same_owner() -> None:
    """The table reads both surfaces; ``losses.reconstruction.lambda_l1`` written by
    the author resolves the same way an ``image_losses`` entry does."""
    s = _bare_for_setup(_weights_cfg([], reconstruction=SimpleNamespace(lambda_l1=3.0)))
    GeoMambaULFStrategy._setup_strategy_specific_components(s)
    assert (s._lambda_l1, s._lambda_ph_w2, s._lambda_beltrami) == (3.0, 0.0, 0.0)


def test_an_arm_without_an_l1_entry_is_refused() -> None:
    s = _bare_for_setup(_weights_cfg([("cubical_ph_w2", 1.0)]))
    with pytest.raises(ValueError, match=r"losses\.image_losses"):
        GeoMambaULFStrategy._setup_strategy_specific_components(s)


def test_a_weighted_beltrami_on_a_one_channel_output_raises() -> None:
    """Planted violation: 13 arms declared the term at out_channels=1 and it never
    computed (the silent skip this replaces)."""
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=lambda x, mask=None, contrast_id=None: x)
    s._ulf_unwarp = None
    s._uncertainty_loss = None
    s._lambda_l1, s._lambda_ph_w2, s._lambda_beltrami = 10.0, 0.0, 0.05
    s._topology_warmup_epochs = 0
    s._ph_loss = None
    s._beltrami_loss = None
    x = torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError, match="displacement_head"):
        GeoMambaULFStrategy._compute_losses_impl(s, x, x, epoch=0)


def test_an_unweighted_beltrami_on_a_one_channel_output_is_simply_absent() -> None:
    """Weight 0 is an absent term, not an error, so every arm that deleted the entry
    trains on a 1-channel head."""
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=lambda x, mask=None, contrast_id=None: x)
    s._ulf_unwarp = None
    s._uncertainty_loss = None
    s._lambda_l1, s._lambda_ph_w2, s._lambda_beltrami = 10.0, 0.0, 0.0
    s._topology_warmup_epochs = 0
    s._ph_loss = None
    s._beltrami_loss = None
    x = torch.rand(1, 1, 8, 8)
    losses = GeoMambaULFStrategy._compute_losses_impl(s, x, x, epoch=0)
    assert set(losses) == {"loss_l1", "g_total_loss"}


def test_loss_ownership_is_declared() -> None:
    """The class states what it computes inline and that it folds nothing; the private
    weight resolver that made the same YAML weigh differently here is gone."""
    assert GeoMambaULFStrategy.inline_losses == frozenset(
        {"l1", "cubical_ph_w2", "beltrami_diagnostic"}
    )
    assert GeoMambaULFStrategy.folds_image_losses is False
    assert not hasattr(GeoMambaULFStrategy, "_read_loss_weight")
    assert not hasattr(GeoMambaULFStrategy, "_search_loss_weight")


def _schema_block(**metric_sfc):
    from spectramr.config.schemas.training.geomamba_ulf import (
        GeoMambaULFTrainingConfigSchema,
    )

    return GeoMambaULFTrainingConfigSchema(**({"metric_sfc": metric_sfc} if metric_sfc else {}))


def test_a_declared_metric_sfc_on_a_model_without_one_raises() -> None:
    """#1752: the block was accepted and silently dropped when the generator has
    no ``sfc_beta``; an arm that writes it must be told nothing applies it."""
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=torch.nn.Conv2d(1, 1, 1))
    with pytest.raises(ValueError, match="no metric space-filling curve"):
        s._apply_metric_sfc_config(_schema_block(beta=0.5))


def test_the_default_filled_block_on_a_model_without_one_stays_silent() -> None:
    """The schema fills ``metric_sfc`` by default, so an arm that never wrote it
    must keep training on a model without a metric SFC."""
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=torch.nn.Conv2d(1, 1, 1))
    s._apply_metric_sfc_config(_schema_block())


def test_get_ph_loss_builds_from_topology_config(monkeypatch) -> None:
    """The declared topology knobs reach the loss, and the applied values are stamped."""
    import spectramr.models.losses.cubical_ph_w2_loss as ph_mod
    from spectramr.config.schemas.training.geomamba_ulf import TopologyConfig

    seen: dict[str, Any] = {}

    class _RecordingPH:
        def __init__(self, homology_dims, wasserstein_p, stratify_subvolumes):
            seen.update(
                homology_dims=homology_dims,
                wasserstein_p=wasserstein_p,
                stratify_subvolumes=stratify_subvolumes,
            )
            self.homology_dims = tuple(homology_dims)
            self.wasserstein_p = wasserstein_p
            self.stratify_subvolumes = stratify_subvolumes

    monkeypatch.setattr(ph_mod, "CubicalPHWassersteinLoss", _RecordingPH)
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s._ph_loss = None
    s._topology_cfg = TopologyConfig(ph_dimensions=[0], wasserstein_p=1)
    s._get_ph_loss()
    assert seen == {"homology_dims": (0,), "wasserstein_p": 1, "stratify_subvolumes": 1}
    stamped = s._topology_provenance["cubical_ph_w2"]
    assert stamped["declared"]["wasserstein_p"] == 1
    assert stamped["applied"]["homology_dims"] == [0]


# ---------------------------------------------------------------------------
# Physical unwarp: one helper for training and validation
# ---------------------------------------------------------------------------


def _unwarp_geo_cfg(declared: bool, **unwarp_kw):
    from spectramr.config.schemas.training.geomamba_ulf import UnwarpConfig

    unwarp_kw.setdefault("enable_grad_nonlinearity", False)
    # An active EPI block must carry its shift parameter since ddcb9ab14.
    unwarp_kw.setdefault("echo_spacing_s", 5e-4)
    unwarp = UnwarpConfig(**unwarp_kw)
    fields = {"unwarp"} if declared else set()
    return SimpleNamespace(unwarp=unwarp, model_fields_set=fields)


def test_undeclared_unwarp_block_builds_no_unwarp() -> None:
    assert GeoMambaULFStrategy._build_ulf_unwarp(_unwarp_geo_cfg(False)) is None


def test_declared_unwarp_block_with_b0_and_no_map_raises() -> None:
    cfg = _unwarp_geo_cfg(True, enable_b0=True, enable_maxwell=False)
    with pytest.raises(ValueError, match="b0_map_key"):
        GeoMambaULFStrategy._build_ulf_unwarp(cfg)


def test_prepare_input_is_the_same_transform_on_every_call() -> None:
    """Train and validation both call _prepare_input, so one batch maps to one input."""
    cfg = _unwarp_geo_cfg(
        True,
        enable_b0=False,
        enable_maxwell=True,
        echo_spacing_s=5.5e-4,
        gradient_amplitudes_mt_per_m=[20.0, 20.0, 20.0],
        field_strength_t=0.064,
    )
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s._ulf_unwarp = GeoMambaULFStrategy._build_ulf_unwarp(cfg)
    x = torch.linspace(0.0, 1.0, 4 * 8 * 8).reshape(1, 1, 4, 8, 8)
    first = s._prepare_input(x, {})
    second = s._prepare_input(x, {})
    assert torch.equal(first, second)
    assert not torch.allclose(first, x)


def test_prepare_input_is_identity_without_unwarp() -> None:
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s._ulf_unwarp = None
    x = torch.rand(1, 1, 4, 8, 8)
    assert s._prepare_input(x, {}) is x


# ---------------------------------------------------------------------------
# Validation carries the batch's contrast and brain mask into the generator
# ---------------------------------------------------------------------------


class _RecordingGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, x, mask=None, contrast_id=None):
        self.calls.append({"mask": mask, "contrast_id": contrast_id})
        return x


def _validation_strategy(monkeypatch, n_contrasts: int):
    gen = _RecordingGenerator()
    s = GeoMambaULFStrategy.__new__(GeoMambaULFStrategy)
    s.env = SimpleNamespace(generator=gen, device=torch.device("cpu"))
    s.device = torch.device("cpu")
    s._film_n_contrasts = n_contrasts
    s._ulf_unwarp = None
    stub = SimpleNamespace(get_metric_names=lambda: ["psnr"])
    monkeypatch.setattr(
        GeoMambaULFStrategy, "validation_metrics_computer", property(lambda _self: stub)
    )
    scored = []

    def _score(self, output, target):
        scored.append((output.shape, target.shape))
        return {"psnr": 1.0}

    monkeypatch.setattr(GeoMambaULFStrategy, "_score_validation_output", _score)
    return s, gen, scored


def test_validation_passes_the_batch_contrast_and_mask(monkeypatch) -> None:
    s, gen, scored = _validation_strategy(monkeypatch, n_contrasts=3)
    mask = torch.ones(1, 1, 4, 4)
    metrics = s.validation_step(
        torch.zeros(1, 1, 4, 4),
        torch.zeros(1, 1, 4, 4),
        batch_data={"contrast_idx": torch.tensor([2]), "foreground_mask": mask},
    )
    assert metrics == {"psnr": 1.0}
    assert gen.calls[0]["contrast_id"].tolist() == [2]
    assert torch.equal(gen.calls[0]["mask"], mask)
    assert len(scored) == 1


def test_validation_raises_without_a_contrast_key_when_film_is_multi_contrast(monkeypatch) -> None:
    s, gen, _ = _validation_strategy(monkeypatch, n_contrasts=3)
    with pytest.raises(ValueError, match="contrast_id/contrast_idx"):
        s.validation_step(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), batch_data={})
    assert gen.calls == []


def test_validation_without_film_needs_no_contrast_key(monkeypatch) -> None:
    s, gen, _ = _validation_strategy(monkeypatch, n_contrasts=1)
    s.validation_step(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    assert gen.calls[0]["contrast_id"] is None
    assert gen.calls[0]["mask"] is None


def test_validation_feeds_the_unwarped_input(monkeypatch) -> None:
    s, _gen, _ = _validation_strategy(monkeypatch, n_contrasts=1)
    seen = []
    monkeypatch.setattr(
        GeoMambaULFStrategy, "_prepare_input", lambda self, x, b: seen.append(x) or x + 1.0
    )
    s.validation_step(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), batch_data={})
    assert len(seen) == 1


def test_an_undeclared_unwarp_block_defaults_to_every_term_off() -> None:
    """The schema default must construct: the all-on field defaults fail the shift check."""
    from spectramr.config.schemas.training.geomamba_ulf import GeoMambaULFTrainingConfigSchema

    unwarp = GeoMambaULFTrainingConfigSchema().unwarp
    assert not (unwarp.enable_b0 or unwarp.enable_grad_nonlinearity or unwarp.enable_maxwell)
