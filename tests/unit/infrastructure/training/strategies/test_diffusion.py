"""DiffusionTrainingStrategy latent-conditioning wiring (2026-07 ldm triage).

The latent-diffusion branch used to (a) never populate ``condition_image`` and
(b) drop ``gen_kwargs`` entirely in ``_forward_through_model`` — so the ULF
condition and contrast context never reached the model and it ran
unconditionally. These pin the fix via unbound calls with a MagicMock strategy.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

from spectramr.config.schemas.enums import PredictionType
from spectramr.data.batch_types import TrainingBatch
from spectramr.data.transforms.normalization import (
    DECOMPRESS_MAGNITUDE_CEILING,
    compress_kspace_log,
    decompress_kspace_log,
)
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.schedulers.diffusion_scheduler import (
    DiffusionScheduler,
)
from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
    validation_loss_key,
)
from spectramr.infrastructure.training.utils.kspace_view import decompress_for_view
from spectramr.infrastructure.training.utils.metric_views import MetricViews


def _gen(conditional: bool):
    return SimpleNamespace(config=SimpleNamespace(conditional_translation=conditional))


def test_build_gen_kwargs_sets_condition_image_when_advertised() -> None:
    mock = MagicMock()
    mock.generator_model = _gen(conditional=True)
    # neutralize the smaps / prior / accel branches
    mock._current_smaps = None
    del mock.prior_model
    mock.config = SimpleNamespace(undersampling=None)
    inp = torch.randn(2, 1, 32, 32)
    kw = DiffusionTrainingStrategy._build_generator_kwargs(
        mock,
        is_cold_diffusion=False,
        is_latent_diffusion=True,
        input_batch=inp,
        target_batch=inp,
        batch_data=None,
    )
    assert "condition_image" in kw and torch.equal(kw["condition_image"], inp)


def test_build_gen_kwargs_no_condition_image_when_unconditional() -> None:
    mock = MagicMock()
    mock.generator_model = _gen(conditional=False)
    mock._current_smaps = None
    del mock.prior_model
    mock.config = SimpleNamespace(undersampling=None)
    inp = torch.randn(2, 1, 32, 32)
    kw = DiffusionTrainingStrategy._build_generator_kwargs(
        mock,
        is_cold_diffusion=False,
        is_latent_diffusion=True,
        input_batch=inp,
        target_batch=inp,
        batch_data=None,
    )
    assert "condition_image" not in kw


@pytest.mark.parametrize("geometry_channels", [6, 0])
def test_build_gen_kwargs_forwards_the_acquired_plane_to_a_consumer(geometry_channels) -> None:
    """The geometry planes read the scanner-written support; the reverse loops pass
    it from ``measured_lines``, so training must pass the same plane or the two
    paths condition on different definitions."""
    mock = MagicMock()
    mock.generator_model = SimpleNamespace(
        config=SimpleNamespace(conditional_translation=False),
        kspace_geometry_channels=geometry_channels,
    )
    mock._current_smaps = None
    del mock.prior_model
    mock.config = SimpleNamespace(undersampling=None)
    inp = torch.randn(1, 2, 16, 16)
    mask = torch.ones(1, 1, 16, 16)
    acquired = torch.zeros(1, 1, 16, 16)
    acquired[..., 2:14] = 1.0
    kw = DiffusionTrainingStrategy._build_generator_kwargs(
        mock,
        is_cold_diffusion=True,
        is_latent_diffusion=False,
        input_batch=inp,
        target_batch=inp,
        batch_data=None,
        mask=mask,
        acquired=acquired,
    )
    if geometry_channels:
        assert kw["acquired"] is acquired
    else:
        assert "acquired" not in kw, "a generator without geometry planes must not receive it"


def test_validation_step_no_acceleration_runs_plain_pass_emits_val_psnr() -> None:
    """Image / latent-translation arms have no ``acceleration`` block. The
    k-space cascade must be skipped: it would AttributeError on
    ``self.config.undersampling.base_acceleration`` (phase 11 renamed the block) and,
    even guarded, only ever emit ``val_*_{R}x`` — never the plain ``val_psnr``
    that ``early_stopping.metric`` selects on (pitfall #18). A single plain pass
    emits unsuffixed metrics via ``_compute_validation_metrics``.
    """
    inp = torch.randn(1, 1, 16, 16)
    tgt = torch.randn(1, 1, 16, 16)

    mock = MagicMock()
    mock.config = SimpleNamespace(
        undersampling=None,
        model=SimpleNamespace(in_channels=1, input_type="image"),
        validation=SimpleNamespace(input_dependence_tol=None),
    )
    mock.num_timesteps = 1000
    mock._prepare_validation_data = MagicMock(return_value=(inp, tgt, torch.tensor(1.0)))
    mock._generate_validation_prediction = MagicMock(return_value=tgt)
    mock._compute_validation_metrics = MagicMock(
        return_value={"val_psnr": 30.0, "val_ssim": 0.9, "val_mae": 0.1}
    )

    out = DiffusionTrainingStrategy.validation_step(mock, inp, tgt)

    assert out["val_psnr"] == 30.0  # unsuffixed — resolves early_stopping.metric
    assert not any(k.endswith(("_2x", "_8x", "_32x")) for k in out)  # no cascade cols
    mock._generate_validation_prediction.assert_called_once()  # single pass


def test_forward_through_model_latent_forwards_condition_and_context() -> None:
    received = {}

    def _spy(model_input, timesteps=None, context=None, condition_image=None):
        received["context"] = context
        received["condition_image"] = condition_image
        return torch.zeros(1)

    mock = MagicMock()
    mock.generator_model = _spy
    cond = torch.randn(1, 1, 16, 16)
    cidx = torch.tensor([2])
    DiffusionTrainingStrategy._forward_through_model(
        mock,
        model_input=torch.randn(1, 4, 8, 8),
        timesteps=torch.tensor([1]),
        is_latent_diffusion=True,
        gen_kwargs={"condition_image": cond, "contrast_idx": cidx},
    )
    assert torch.equal(received["condition_image"], cond)
    assert received["context"] == {"contrast_idx": cidx}


# ---------------------------------------------------------------------------
# Validation-path contrast conditioning (#1931).
#
# The train path was fixed in half 1: ``_contrast_idx_from_batch`` is the ONE
# owner of the extraction (non-negotiable 17). ``_generate_validation_prediction``
# kept a SECOND resolver behind ``isinstance(batch_data, dict)`` -- and what
# ``select_validation_extra_fields`` (pipelines/train.py) forwards as
# ``batch_data`` is a ``TrainingBatch``: a dataclass, not a mapping, whose
# non-core fields live in ``.metadata``. So the guard never matched and every
# multi-contrast arm trained CONDITIONED and validated UNCONDITIONED -- the
# train/val mismatch of pitfall #18, silent, with the checkpoint selected on
# metrics the model was never conditioned for.
#
# These drive the REAL method rather than the extraction in isolation, because
# the whole body sits inside ``except Exception: return None`` -- an assertion
# on a helper would not notice the swallow.
# ---------------------------------------------------------------------------


def _validation_strategy(chunk_size: int = 8):
    """A strategy mock that reaches the plain ``_forward_through_model`` branch.

    Two ``MagicMock`` hazards are deliberately avoided here:

    * ``hasattr(mock, anything)`` is always True, so a ``MagicMock`` generator
      would satisfy ``hasattr(gen, "sample")`` and route into the latent
      sampler. ``generator_model`` is therefore a real bare object.
    * ``mock._contrast_idx_from_batch(...)`` would return a ``Mock`` -- truthy,
      forwarded, and the test would pass whether or not the extraction works.
      The REAL staticmethod is bound instead, which is the whole subject.
    """
    mock = MagicMock()
    mock.generator_model = SimpleNamespace()  # no .sample/.generate/.module
    mock._is_cold_diffusion.return_value = False
    mock._is_latent_diffusion.return_value = False
    mock._contrast_idx_from_batch = DiffusionTrainingStrategy._contrast_idx_from_batch
    mock.config = SimpleNamespace(
        model=SimpleNamespace(model_type="diffusion_unet"),  # skips the smaps block
        validation=SimpleNamespace(loader=SimpleNamespace(chunk_size=chunk_size)),
        training=None,  # the legacy-forward branch reads config.training.diffusion
    )
    mock.q_sample.side_effect = lambda x, t, noise: x
    mock._forward_through_model.return_value = torch.zeros(1, 1, 8, 8)
    return mock


def _drive_validation(mock, batch_data, *, target: torch.Tensor | None = None):
    """Run the real method and fail loudly if its blanket except swallowed."""
    inp = torch.randn(1, 1, 8, 8)
    tgt = target if target is not None else torch.zeros(1, 1, 8, 8)
    mock._forward_through_model.return_value = torch.zeros_like(tgt)
    DiffusionTrainingStrategy._generate_validation_prediction(
        mock,
        input_batch=inp,
        target_batch=tgt,
        timestep=torch.tensor([1]),
        batch_data=batch_data,
        kwargs={},
    )
    swallowed = [
        c.args[0]
        for c in mock.logging_service.log_warning.call_args_list
        if c.args and "Validation generation failed" in str(c.args[0])
    ]
    assert not swallowed, f"the method's blanket except fired: {swallowed}"
    assert mock._forward_through_model.called, "never reached the generator forward"
    return mock._forward_through_model.call_args.kwargs["gen_kwargs"]


def test_validation_conditions_on_a_training_batch_not_only_a_dict() -> None:
    """The load-bearing test: a ``TrainingBatch`` is what production delivers."""
    c_idx = torch.tensor([2])
    batch = TrainingBatch(
        input=torch.randn(1, 1, 8, 8),
        target=torch.zeros(1, 1, 8, 8),
        metadata={"contrast_idx": c_idx},
    )
    assert not isinstance(batch, dict)  # the guard that used to stand here

    gen_kwargs = _drive_validation(_validation_strategy(), batch)

    assert "contrast_idx" in gen_kwargs, (
        "validation reached the generator with no contrast_idx while the train "
        "path sends one — the arm validates unconditioned (pitfall #18)"
    )
    assert torch.equal(gen_kwargs["contrast_idx"], c_idx)


def test_validation_still_conditions_on_a_plain_dict_batch() -> None:
    """Routing through the owner must not drop the shape that DID work."""
    c_idx = torch.tensor([1])
    gen_kwargs = _drive_validation(
        _validation_strategy(),
        {"input": torch.randn(1, 1, 8, 8), "contrast_idx": c_idx},
    )
    assert torch.equal(gen_kwargs["contrast_idx"], c_idx)


def test_validation_realigns_contrast_ids_when_5d_targets_are_flattened() -> None:
    """This site's own concern, kept out of the owner: only validation flattens.

    Two volumes of three slices arrive as one batch of six, so a per-volume
    contrast id must be repeated per slice. Getting this wrong is the silent
    shape: ``repeat`` instead of ``repeat_interleave`` labels slice 0 of volume
    1 with volume 0's contrast and still trains without error.
    """
    c_idx = torch.tensor([0, 2])
    batch = TrainingBatch(
        input=torch.randn(2, 1, 3, 8, 8),
        target=torch.zeros(2, 1, 3, 8, 8),
        metadata={"contrast_idx": c_idx},
    )
    gen_kwargs = _drive_validation(_validation_strategy(), batch, target=torch.zeros(6, 1, 8, 8))
    assert torch.equal(gen_kwargs["contrast_idx"], torch.tensor([0, 0, 0, 2, 2, 2]))


def test_validation_sends_no_contrast_idx_when_the_batch_carries_none() -> None:
    """A single-contrast arm must not gain a fabricated id (#9)."""
    batch = TrainingBatch(input=torch.randn(1, 1, 8, 8), target=torch.zeros(1, 1, 8, 8))
    gen_kwargs = _drive_validation(_validation_strategy(), batch)
    assert "contrast_idx" not in gen_kwargs


# ---------------------------------------------------------------------------
# Reverse-schedule SSOT (2026-07-11 stage-2 LDM triage).
#
# The strategy q_samples with training.diffusion.noise_schedule, but
# LatentDiffusionGenerator built its OWN reverse Diffusion from the dataclass
# default beta_schedule="linear" that no YAML wires. An arm declaring
# `noise_schedule: cosine` therefore trained on a cosine forward trajectory and
# inverted it with linear posterior coefficients at validation → the decoded
# image collapsed to ~black (stage-2 LDM: train_psnr≈32, val_psnr≈6 dB).
# ---------------------------------------------------------------------------


def _real_ldm(**kw):
    from spectramr.models.generators.latent_diffusion_generator import (
        LatentDiffusionGenerator,
    )

    return LatentDiffusionGenerator(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        timesteps=5,
        device="cpu",
        num_layers=3,
        **kw,
    )


def test_bind_reverse_schedule_syncs_ldm_to_training_ssot() -> None:
    gen = _real_ldm()
    assert gen.beta_schedule == "linear"  # the un-wired dataclass default

    mock = MagicMock()
    mock._is_latent_diffusion.return_value = True
    mock.generator_model = gen
    mock.num_timesteps = 7
    mock.beta_schedule = "cosine"
    mock.device = "cpu"
    # The strategy now hands over the forward process's ACTUAL betas rather than
    # only its name, so the mock must carry a real forward scheduler.
    mock.scheduler = DiffusionScheduler(num_timesteps=7, beta_schedule="cosine")

    DiffusionTrainingStrategy._bind_generator_reverse_schedule(mock)

    assert gen.beta_schedule == "cosine"
    assert gen.timesteps == 7
    assert gen.diffusion.timesteps == 7
    # betas must now come from the cosine schedule, not the linear one.
    assert not torch.allclose(gen.diffusion.betas[:5], _real_ldm().diffusion.betas[:5])
    # ...and they must be the FORWARD process's betas exactly, not the
    # generator's own re-derivation of a schedule that merely shares the name.
    torch.testing.assert_close(gen.diffusion.betas.cpu(), mock.scheduler.betas.cpu())


def test_bind_reverse_schedule_noop_for_non_latent_diffusion() -> None:
    mock = MagicMock()
    mock._is_latent_diffusion.return_value = False
    # Must return without touching the generator at all.
    DiffusionTrainingStrategy._bind_generator_reverse_schedule(mock)
    mock.logging_service.log_info.assert_not_called()


def test_bind_reverse_schedule_raises_when_model_cannot_sync() -> None:
    """A generator that owns a private reverse schedule but exposes no way to
    sync it must RAISE — sampling on a schedule that disagrees with training is
    exactly the silent divergence CLAUDE.md #9 forbids."""
    import pytest

    mock = MagicMock()
    mock._is_latent_diffusion.return_value = True
    # has `.diffusion`, but no `set_diffusion_schedule`
    mock.generator_model = SimpleNamespace(diffusion=object())

    with pytest.raises(RuntimeError, match="set_diffusion_schedule"):
        DiffusionTrainingStrategy._bind_generator_reverse_schedule(mock)


def test_diffusion_reads_config_directly_not_via_getattr_fallback() -> None:
    """SSOT: diffusion.py must not read the frozen config via getattr-fallback.

    This is a Phase-2.3 "FIXED" file that had regressed — ``getattr(self.config.*,
    ...)`` and nested ``getattr(getattr(self.config, ...))`` fallbacks crept back in
    (issue #368). All were converted to guarded direct access (the schema
    guarantees every declared section/field; ``| None`` sections carry an explicit
    ``is not None`` guard). Pin it here so a future edit cannot silently reintroduce
    a config getattr on this file: the global test_ssot_compliance guard is
    CI-advisory and let it drift once already.
    """
    import ast
    import inspect

    import spectramr.infrastructure.training.strategies.diffusion as diff

    tree = ast.parse(inspect.getsource(diff))
    offenders: list[tuple[int, str]] = []
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "getattr"
            and call.args
        ):
            continue
        base = ast.unparse(call.args[0])
        # The SSOT config object and its attribute chains -- NOT a model's own
        # ``gen.config`` (base ``gen``) or ``getattr(self, "config", None)`` (base
        # ``self``), both of which are legitimately dynamic.
        if base in {"config", "self.config"} or base.startswith(("config.", "self.config.")):
            offenders.append((call.lineno, ast.unparse(call)[:70]))
    assert not offenders, (
        "diffusion.py reads the SSOT config via getattr-fallback; use guarded "
        f"direct access instead: {offenders}"
    )


class TestSamplerStepParamNames:
    """``_SAMPLER_STEP_PARAM_NAMES`` — the validation.sampler_steps forwarding seam.

    ``_generate_validation_prediction`` forwards the resolved step count only
    under a parameter name the sampler's ``sample()`` actually exposes. When the
    probe matches nothing the count cannot be honoured, so the strategy must say
    so rather than drop it — the ldm_ulf_to_hf stage-2 arms pinned
    ``sampler_steps: 25`` and silently ran the full 1000-step chain instead.
    """

    def test_probe_order_puts_the_diffusers_name_first(self):
        from spectramr.infrastructure.training.strategies.diffusion import (
            _SAMPLER_STEP_PARAM_NAMES,
        )

        assert _SAMPLER_STEP_PARAM_NAMES[0] == "num_inference_steps"
        assert len(set(_SAMPLER_STEP_PARAM_NAMES)) == len(_SAMPLER_STEP_PARAM_NAMES)

    def test_latent_diffusion_sampler_exposes_a_probed_name(self):
        # The in-repo latent-diffusion sampler must stay forwardable; this is the
        # regression guard for the silent-drop bug.
        import inspect

        from spectramr.infrastructure.training.strategies.diffusion import (
            _SAMPLER_STEP_PARAM_NAMES,
        )
        from spectramr.models.generators.latent_diffusion_generator import (
            LatentDiffusionGenerator,
        )

        params = inspect.signature(LatentDiffusionGenerator.sample).parameters
        matched = [n for n in _SAMPLER_STEP_PARAM_NAMES if n in params]
        assert matched, (
            "LatentDiffusionGenerator.sample() exposes none of "
            f"{_SAMPLER_STEP_PARAM_NAMES}; validation.sampler_steps would be dropped."
        )


# ---------------------------------------------------------------------------
# Timestep-sampling floor (issue #535)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Consistency-model validation steps: the declared count reaches multistep_inference.
# ---------------------------------------------------------------------------


class _StepsDiffusion:
    """``training.diffusion`` stand-in with pydantic's ``model_fields_set``."""

    def __init__(self, declared: dict) -> None:
        self.sampling_steps = declared.get("sampling_steps", 100)
        self.model_fields_set = set(declared)


def _steps_strategy(val_steps=None, diffusion=None):
    st = object.__new__(DiffusionTrainingStrategy)
    st.num_timesteps = 1000
    st.config = SimpleNamespace(
        validation=SimpleNamespace(sampling=SimpleNamespace(steps=val_steps)),
        training=SimpleNamespace(diffusion=_StepsDiffusion(diffusion or {})),
    )
    return st


@pytest.mark.parametrize(
    ("val_steps", "diffusion", "expected"),
    [
        (None, {}, 2),  # undeclared: the historical 2, not the schema's 100
        (8, {}, 8),
        (None, {"sampling_steps": 1}, 1),
        (4, {"sampling_steps": 1}, 4),  # validation.sampling.steps wins
    ],
)
def test_consistency_validation_steps_follow_the_declaration(val_steps, diffusion, expected) -> None:
    assert _steps_strategy(val_steps, diffusion)._consistency_validation_steps() == expected


def test_the_declared_steps_reach_the_consistency_sampler() -> None:
    from spectramr.models.generators.consistency_model_generator import ConsistencyModelGenerator

    torch.manual_seed(0)
    gen = ConsistencyModelGenerator(in_channels=1, out_channels=1, features=[8, 16], time_dim=16).eval()
    calls: list[int] = []
    real = gen.multistep_inference

    def _spy(x, steps=2, **kw):
        calls.append(steps)
        return real(x, steps=steps, **kw)

    gen.multistep_inference = _spy  # type: ignore[method-assign]
    mock = _validation_strategy()
    mock.generator_model = gen
    mock._consistency_validation_steps.return_value = 5
    mock.config.training = SimpleNamespace(diffusion=SimpleNamespace(inference_sampler=None))
    inp = torch.randn(1, 1, 8, 8)
    out = DiffusionTrainingStrategy._generate_validation_prediction(
        mock, input_batch=inp, target_batch=torch.zeros(1, 1, 8, 8),
        timestep=torch.tensor([1]), batch_data=None, kwargs={},
    )
    assert calls == [5]
    torch.manual_seed(0)
    two = real(inp, steps=2)
    assert not torch.allclose(out, two)


def _timestep_strategy(
    sampling_strategy: str,
    floor: int,
    num_timesteps: int = 28,
    *,
    curriculum_start: int | None = 4,
    curriculum_rate: float | None = 0.005,
    max_iterations: int = 30000,
):
    """Unbound-call harness for ``sample_timesteps``.

        The method only needs ``generator_model.training``, the resolved curriculum
        state, the sampling-strategy name, ``num_timesteps``, ``device`` and the
        operator floor — so a MagicMock strategy exercises it without building a
        model.

    The curriculum is not hand-set (#1296): the fixture binds the REAL
        ``_resolve_curriculum_once`` so the state is derived from the same config
        declared two lines below it. A fixture that stated the answer independently
        could disagree with the config beside it, which is the shape that lets a
        mock-fed test pass while production does something else -- and here it would
        also stub out the very resolver these cases exist to exercise.
    """
    strategy = MagicMock(spec=DiffusionTrainingStrategy)
    strategy.generator_model = SimpleNamespace(training=True)
    strategy.num_timesteps = num_timesteps
    strategy.device = torch.device("cpu")
    strategy.config = SimpleNamespace(
        training=SimpleNamespace(
            curriculum_start_timestep=curriculum_start,
            curriculum_ramp_rate=curriculum_rate,
            max_iterations=max_iterations,
            timestep_sampling_strategy=sampling_strategy,
        )
    )
    strategy._curriculum_state = None
    strategy.logging_service = MagicMock()
    strategy._resolve_curriculum_once = MethodType(
        DiffusionTrainingStrategy._resolve_curriculum_once, strategy
    )
    strategy._min_meaningful_timestep = lambda: floor
    return strategy


# ── the short-run bypass (#1296) ────────────────────────────────────────────
#
# `test_declared_curriculum_still_caps_the_timestep_range` and its neighbours
# below already cover declared / absent / half-declared. The case none of them
# reached is the third one: a curriculum that IS fully declared and still does
# not run, because `max_iterations` is at or below the short-run bypass.


def test_the_short_run_bypass_uses_the_full_range_despite_a_declared_ramp() -> None:
    """The behaviour #1296 documents rather than changes.

    A <=5k-iteration run ignores the ramp entirely -- correct, because the ramp
    would never reach meaningful acceleration in that many steps. The point is
    that the arm still DECLARED one, which the curriculum state now records so
    the strategy can warn instead of leaving the two cases indistinguishable.
    """
    strategy = _timestep_strategy("uniform", floor=0, max_iterations=2000)
    state = strategy._resolve_curriculum_once()
    assert not state.effective
    assert state.declared
    strategy.logging_service.log_warning.assert_called_once()
    drawn = DiffusionTrainingStrategy.sample_timesteps(strategy, batch_size=4096, iteration=200)
    assert drawn.max().item() == 27
    # Resolved ONCE: the sampler above must reuse the cached state rather than
    # re-reading the config every step (non-negotiable 9), and the latched
    # warning is the observable proof that it did.
    strategy.logging_service.log_warning.assert_called_once()


ALL_SAMPLERS = [
    "uniform",
    "importance",
    "linear_decay",
    "high_t_emphasis",
    "balanced_high_t",
]


@pytest.mark.parametrize("sampler", ALL_SAMPLERS)
def test_sampler_can_draw_the_operator_floor(sampler: str) -> None:
    """Every strategy must give the floor timestep positive mass.

    The reverse loop's final step runs at the floor and its full reveal writes
    54-99% of the reconstruction. Before the fix every strategy drew from
    ``[1, high)`` and ``high_t_emphasis`` weighted t by t, so the floor had
    literally zero probability while validation ended there every time.
    """
    strategy = _timestep_strategy(sampler, floor=0)
    drawn = DiffusionTrainingStrategy.sample_timesteps(strategy, batch_size=4096, iteration=30000)
    assert drawn.min().item() >= 0
    assert 0 in set(drawn.tolist()), f"{sampler} never drew the floor timestep"


@pytest.mark.parametrize("sampler", ALL_SAMPLERS)
def test_sampler_respects_a_nonzero_floor(sampler: str) -> None:
    """When t=0 is the identity task the floor is 1 and must be honoured."""
    strategy = _timestep_strategy(sampler, floor=1)
    drawn = DiffusionTrainingStrategy.sample_timesteps(strategy, batch_size=2048, iteration=30000)
    assert drawn.min().item() >= 1


@pytest.mark.parametrize("sampler", ALL_SAMPLERS)
def test_sampler_stays_inside_the_timestep_range(sampler: str) -> None:
    strategy = _timestep_strategy(sampler, floor=0)
    drawn = DiffusionTrainingStrategy.sample_timesteps(strategy, batch_size=1024, iteration=30000)
    assert drawn.max().item() < 28


def test_balanced_high_t_still_tilts_toward_high_t() -> None:
    """The floor must not flatten the distribution the strategy exists to create."""
    strategy = _timestep_strategy("balanced_high_t", floor=0)
    drawn = DiffusionTrainingStrategy.sample_timesteps(strategy, batch_size=20000, iteration=30000)
    values = drawn.float()
    high_share = (values >= 16).float().mean().item()
    low_share = (values <= 3).float().mean().item()
    assert high_share > 0.45, high_share
    assert low_share > 0.01, low_share
    assert high_share > low_share


def test_unknown_sampler_raises() -> None:
    from spectramr.domain.exceptions import ConfigurationError

    strategy = _timestep_strategy("no_such_sampler", floor=0)
    with pytest.raises(ConfigurationError, match="no_such_sampler"):
        DiffusionTrainingStrategy.sample_timesteps(strategy, batch_size=8, iteration=1)


# --------------------------------------------------------------------------
# training.diffusion.prediction_type wiring (2026-07 ldm cohort review)
#
# The knob had a schema field, a PredictionType enum and 162 declaring arms,
# and ZERO readers -- an arm could ask for v_prediction and silently train
# something else (pitfall #15). The objective is not selectable here: it is
# decided by the path (latent -> epsilon, pixel -> sample), so the wiring is an
# agreement check that raises when the declaration is not what runs.
# --------------------------------------------------------------------------


def _strategy_declaring(prediction_type: str, model_type: str):
    mock = MagicMock()
    mock.config = SimpleNamespace(
        model=SimpleNamespace(model_type=model_type),
        training=SimpleNamespace(diffusion=SimpleNamespace(prediction_type=prediction_type)),
    )
    # Bind the real collaborators — a bare MagicMock would make every
    # _is_latent_diffusion() call truthy and the assertions vacuous.
    mock._is_latent_diffusion = lambda: DiffusionTrainingStrategy._is_latent_diffusion(mock)
    mock._effective_prediction_type = lambda: DiffusionTrainingStrategy._effective_prediction_type(
        mock
    )
    return mock


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("latent_diffusion", "epsilon"),
        ("latent_gaussian_diffusion", "epsilon"),
        ("ldm", "epsilon"),
        ("latent_cold_diffusion", "epsilon"),
        ("kspace_cold_diffusion", "sample"),
        ("enhanced_deep_unet", "sample"),
    ],
)
def test_effective_prediction_type_mirrors_the_loss_target(model_type, expected) -> None:
    """compute_losses targets the latent noise for LDM and target_batch otherwise."""
    mock = _strategy_declaring(expected, model_type)
    got = DiffusionTrainingStrategy._effective_prediction_type(mock)
    assert got == expected


@pytest.mark.parametrize(
    ("model_type", "declared"),
    [("latent_diffusion", "epsilon"), ("ldm", "epsilon")],
)
def test_agreeing_declaration_is_accepted(model_type, declared) -> None:
    mock = _strategy_declaring(declared, model_type)
    DiffusionTrainingStrategy._validate_prediction_type(mock)  # must not raise


@pytest.mark.parametrize("declared", ["epsilon", "sample", "v_prediction"])
def test_pixel_space_path_defers_rather_than_crashing(declared) -> None:
    """Scoped to latent diffusion on purpose.

    9 arms on this strategy declare `epsilon` on the pixel path, whose target is
    target_batch (x0). That is a real defect (issue #641), but whether the YAML
    or the path is wrong is those cohorts' decision -- the check must not turn it
    into a startup crash as a side effect of an LDM review. When #641 is settled,
    this test should be inverted to expect a raise.
    """
    mock = _strategy_declaring(declared, "enhanced_deep_unet")
    DiffusionTrainingStrategy._validate_prediction_type(mock)  # must not raise


@pytest.mark.parametrize(
    ("model_type", "declared"),
    [
        # An LDM arm claiming x0-prediction while the path trains epsilon.
        ("latent_diffusion", "sample"),
        ("latent_gaussian_diffusion", "sample"),
        # Never implemented anywhere on this strategy.
        ("latent_diffusion", "v_prediction"),
    ],
)
def test_contradicting_declaration_raises(model_type, declared) -> None:
    mock = _strategy_declaring(declared, model_type)
    with pytest.raises(ConfigurationError, match="prediction_type"):
        DiffusionTrainingStrategy._validate_prediction_type(mock)


def test_enum_valued_declaration_is_unwrapped() -> None:
    """The schema stores a PredictionType enum, not a bare string."""
    mock = _strategy_declaring(PredictionType.EPSILON, "latent_diffusion")
    DiffusionTrainingStrategy._validate_prediction_type(mock)  # must not raise


# --------------------------------------------------------------------------
# Curriculum knobs are DECLARED `| None` fields, so `hasattr` is always True
# and the `else 50` / `else 0.01` fallbacks were dead code. Any arm that did
# not opt into the curriculum sampled timesteps with start_t=None and died on
# its FIRST training step. Both ldm_two_stage_ulf_to_hf stage-2 arms are in
# that class (neither declares the knobs; max_iterations=200000 skips the
# <=5000 short-run bypass).
# --------------------------------------------------------------------------


def _sampling_strategy(start_t, rate, max_iterations=200000, num_timesteps=1000):
    mock = MagicMock()
    mock.config = SimpleNamespace(
        training=SimpleNamespace(
            curriculum_start_timestep=start_t,
            curriculum_ramp_rate=rate,
            max_iterations=max_iterations,
        )
    )
    mock.generator_model = SimpleNamespace(training=True)
    mock.num_timesteps = num_timesteps
    # Real collaborator returns an int floor; MagicMock would poison the max().
    mock._min_meaningful_timestep.return_value = 0
    # Derived from the config above by the REAL resolver, not hand-set (#1296).
    # A bare MagicMock would hand back a truthy auto-attribute for either the
    # cached state or the resolver method, and silently take the curriculum
    # branch with a Mock start timestep.
    mock._curriculum_state = None
    mock._resolve_curriculum_once = MethodType(
        DiffusionTrainingStrategy._resolve_curriculum_once, mock
    )
    mock.device = torch.device("cpu")
    return mock


def test_absent_curriculum_does_not_crash_first_step() -> None:
    """Regression: this raised TypeError on the pre-fix code."""
    mock = _sampling_strategy(start_t=None, rate=None)
    t = DiffusionTrainingStrategy.sample_timesteps(mock, batch_size=4, iteration=1)
    assert t.shape == (4,)
    assert int(t.min()) >= 0
    assert int(t.max()) < 1000


def test_absent_curriculum_samples_the_full_range() -> None:
    """Schema: 'When None the strategy defaults to no curriculum.'"""
    mock = _sampling_strategy(start_t=None, rate=None)
    seen = torch.cat(
        [
            DiffusionTrainingStrategy.sample_timesteps(mock, batch_size=256, iteration=1)
            for _ in range(8)
        ]
    )
    # A curriculum cap at iteration 1 would pin every draw below ~50.
    assert int(seen.max()) > 500


@pytest.mark.parametrize(("start_t", "rate"), [(50, None), (None, 0.01)])
def test_partially_declared_curriculum_is_treated_as_absent(start_t, rate) -> None:
    """Half a curriculum is not a curriculum -- and must not crash."""
    mock = _sampling_strategy(start_t=start_t, rate=rate)
    t = DiffusionTrainingStrategy.sample_timesteps(mock, batch_size=4, iteration=1)
    assert t.shape == (4,)


def test_declared_curriculum_still_caps_the_timestep_range() -> None:
    """The opt-in path must keep working: a low cap keeps draws small."""
    mock = _sampling_strategy(start_t=50, rate=0.01)
    seen = torch.cat(
        [
            DiffusionTrainingStrategy.sample_timesteps(mock, batch_size=128, iteration=10)
            for _ in range(4)
        ]
    )
    # dynamic_max = 50 + 10*0.01 -> 50, so nothing may exceed it.
    assert int(seen.max()) < 50


def test_reverse_schedule_binding_hands_over_betas_not_just_the_name() -> None:
    """The forward process must own the schedule, not merely name it.

    Binding by name left the generator rebuilding `cosine` from the
    Nichol-Dhariwal s=0.008 formula while training used DiffusionScheduler's
    s=0 formula -- the same desync this binding exists to remove.
    """
    mock = MagicMock()
    mock._is_latent_diffusion.return_value = True
    mock.num_timesteps = 200
    mock.beta_schedule = "cosine"
    mock.device = torch.device("cpu")
    mock.scheduler = SimpleNamespace(betas=torch.linspace(0.01, 0.2, 200))

    gen = MagicMock()
    mock.generator_model = gen
    del gen.module  # exercise the non-DDP branch

    DiffusionTrainingStrategy._bind_generator_reverse_schedule(mock)

    gen.set_diffusion_schedule.assert_called_once()
    kwargs = gen.set_diffusion_schedule.call_args.kwargs
    assert kwargs["timesteps"] == 200
    assert kwargs["beta_schedule"] == "cosine"
    torch.testing.assert_close(kwargs["betas"], mock.scheduler.betas)


# ---------------------------------------------------------------------------
# #927 -- validation image logging must not be told a domain that is not true
# ---------------------------------------------------------------------------
#
# ``_compute_validation_metrics`` set ``is_preds_image = True`` unconditionally
# after ``_apply_metric_transforms``, which has four paths that hand back their
# input untouched. The flag reaches ``kspace_to_image(already_image=...)``, a
# nested guard inside ``_log_validation_images_to_tensorboard``; told
# ``already_image=True`` about an even-channel real tensor it raises, and the
# caller turns that into "Failed to log validation images".
#
# experiment_11_attention_none, cluster run 2026-08-08: 135 such failures and
# ZERO images written, while PSNR over the unconverted k-space read 58 dB and
# robust_mri_psnr went NaN. The flag is now DERIVED via
# ``domain_inference.metric_transform_produced_image``.
#
# There is no seam on the derivation itself -- it is a local in a ~400-line
# method, and ``kspace_to_image`` is nested inside another. These drive the
# logger directly, which is what the derived flag feeds, and pin both polarities
# so a revert to ``= True`` reintroduces a case one of them covers.


def _image_logger_mock():
    mock = MagicMock()
    mock.metrics_service.save_images_batch = MagicMock(return_value=([], []))
    # Real passthroughs: a MagicMock return from either of these reaches
    # torch.is_complex, and the method's own error path swallows the TypeError
    # before the guard under test is ever consulted.
    mock._resolve_federated_target_start = lambda *a, **k: None
    return mock


def _warned_about_already_image(mock) -> bool:
    return any(
        "already_image=True" in str(c.args[0]) if c.args else False
        for c in mock.logging_service.log_warning.call_args_list
    )


@pytest.mark.unit
def test_kspace_flagged_as_image_is_refused_by_the_guard() -> None:
    """The pre-#927 flag value, on the exact tensor from that run.

    8 real channels is what ``_apply_metric_transforms`` returns when it
    no-ops. Claiming it is already an image is the lie the guard exists to
    catch -- if this stops warning, the guard has been weakened.
    """
    mock = _image_logger_mock()
    kspace = torch.zeros(2, 8, 32, 32)

    DiffusionTrainingStrategy._log_validation_images_to_tensorboard(
        mock,
        kspace,
        kspace.clone(),
        kspace.clone(),
        {},
        batch_idx=0,
        is_image_domain=True,
    )

    assert _warned_about_already_image(mock)
    mock.metrics_service.save_images_batch.assert_not_called()


@pytest.mark.unit
def test_unconverted_kspace_still_renders_when_the_flag_is_derived() -> None:
    """The post-#927 value: ``False`` lets the visualiser do the transform.

    This is the payoff -- the same tensor that produced zero images now reaches
    ``save_images_batch``.
    """
    mock = _image_logger_mock()
    kspace = torch.zeros(2, 8, 32, 32)

    DiffusionTrainingStrategy._log_validation_images_to_tensorboard(
        mock,
        kspace,
        kspace.clone(),
        kspace.clone(),
        {},
        batch_idx=0,
        is_image_domain=False,
    )

    assert not _warned_about_already_image(mock)
    mock.metrics_service.save_images_batch.assert_called_once()


@pytest.mark.unit
def test_the_derivation_the_strategy_now_uses_rejects_a_noop_transform() -> None:
    """Bind the call site to the helper on #927's real shape.

    ``_compute_validation_metrics`` passes ``(hr_fakes_for_metrics,
    pred_transformed)``; a no-op returns the first object as the second.
    """
    from spectramr.infrastructure.training.utils.domain_inference import (
        metric_transform_produced_image,
    )

    hr_fakes = torch.zeros(36, 8, 256, 256)

    assert metric_transform_produced_image(hr_fakes, hr_fakes) is False
    assert metric_transform_produced_image(hr_fakes, torch.zeros(36, 1, 256, 256)) is True


class TestNonFiniteSamplerOutputIsReported:
    """A diverged sampler must be named where it happens, not at the PNG.

    ``_generate_validation_prediction`` wraps its body in ``except Exception``,
    but a sampler that returns NaN does not RAISE — it returns a tensor. The
    failure therefore travelled silently to ``MetricsTracker._normalize_images``,
    which rendered it as a solid-black PNG indistinguishable from a legitimate
    render (pitfall #16). The 2026-08-17 experiment_11 run wrote 24 such PNGs
    across three acceleration rungs with nothing in the log to explain them.

    The multistep cold path makes this easy to hit: ``sampling_steps or
    num_timesteps`` means an arm leaving ``validation.sampling.steps`` unset
    runs the full 1000-step reverse loop, and non-finites accumulate.
    """

    @pytest.mark.unit
    def test_a_finite_prediction_reports_nothing(self) -> None:
        from spectramr.infrastructure.training.strategies.diffusion import (
            describe_nonfinite_prediction,
        )

        assert describe_nonfinite_prediction(torch.randn(2, 8, 16, 16)) is None

    @pytest.mark.unit
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_single_bad_value_is_reported_with_its_count(self, bad: float) -> None:
        """One entry in half a million is the real ratio, and it must not be
        rounded away — on a k-space arm that single value blacks out the whole
        batch through the IFFT."""
        from spectramr.infrastructure.training.strategies.diffusion import (
            describe_nonfinite_prediction,
        )

        hr_fakes = torch.randn(2, 8, 16, 16)
        hr_fakes[0, 0, 0, 0] = bad

        report = describe_nonfinite_prediction(hr_fakes)
        assert report is not None
        assert f"1/{hr_fakes.numel()}" in report
        assert "finite magnitude range" in report

    @pytest.mark.unit
    def test_a_fully_diverged_prediction_says_so_instead_of_dividing(self) -> None:
        """With no finite values there is no range to quote; the message must
        still be produced rather than raising on an empty reduction."""
        from spectramr.infrastructure.training.strategies.diffusion import (
            describe_nonfinite_prediction,
        )

        report = describe_nonfinite_prediction(torch.full((2, 8, 4, 4), float("nan")))
        assert report is not None
        assert "no finite values at all" in report

    @pytest.mark.unit
    def test_a_complex_kspace_prediction_is_checked_on_both_parts(self) -> None:
        """Cold-diffusion samplers may return complex k-space; a NaN in the
        imaginary part alone is just as fatal after the IFFT."""
        from spectramr.infrastructure.training.strategies.diffusion import (
            describe_nonfinite_prediction,
        )

        hr_fakes = torch.randn(2, 4, 8, 8, dtype=torch.complex64)
        assert describe_nonfinite_prediction(hr_fakes) is None

        hr_fakes[0, 0, 0, 0] = complex(1.0, float("nan"))
        assert describe_nonfinite_prediction(hr_fakes) is not None


# ---------------------------------------------------------------------------
# The degraded model input has to be visible in a run of ordinary length.
#
# `_prepare_diffusion_inputs` writes the ONLY snapshot carrying the tensor the
# model is actually fed: `noisy_kspace = q_sample(x, t) = x * mask`, the
# accelerated, zero-filled k-space. It was gated on
# `self._cached_log_interval * 5` — i.e. `logging.intervals.log`, a *logging*
# knob that experiment_11 sets to 5000 — putting it at step 25 000. It never
# fired, and the only remaining "input" artifact was the base strategy's
# `first_steps/input_prepared`, which `_prepare_model_input` leaves untouched
# whenever the data is already in the model's domain. Cold-diffusion runs
# therefore looked like they were being trained on fully-sampled data.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_cold_snapshot_gate_is_not_derived_from_the_logging_interval() -> None:
    """Bind the call site: it declares UNCONDITIONALLY and gates on nothing.

    Source-level because the alternative is running a full cold-diffusion step;
    the invariant being pinned is precisely a *textual* one — this method must
    not reach for a cadence knob again.

    The bar rose when the emission became a declaration. It is no longer enough
    to use the *right* cadence helper here: the wrapper's contract check is
    deliberately not step-dependent, so ANY gate at this site makes the two
    disagree. Under `snapshots.interval_steps: 100` a `snapshot_step_is_due`
    gate would declare on step 0 and let the wrapper raise on step 1 — a
    violation manufactured by the gate itself. Cadence belongs to
    `save_debug_snapshot`, which is the only thing that can see the budget too.
    """
    import inspect

    # Comment-blind: the fix leaves a comment naming the retired expression, and
    # a raw substring search would match the very prose explaining the defect.
    src = "\n".join(
        line
        for line in inspect.getsource(
            DiffusionTrainingStrategy._prepare_diffusion_inputs
        ).splitlines()
        if not line.lstrip().startswith("#")
    )

    assert "self._declare_model_input(" in src, (
        "the degraded model input must be declared, not emitted here — an "
        "overriding _compute_losses_impl silently drops an emission (the seven "
        "diffusion subclasses did exactly that)"
    )
    assert "snapshot_step_is_due" not in src, (
        "the declaration is gated on a cadence again; the wrapper's check is "
        "not step-dependent, so any gate here manufactures false violations"
    )
    # Narrow on purpose: this method's `logger.debug` cadence reads
    # `_cached_log_interval` and is RIGHT to — that one really is a log. Only the
    # snapshot's borrowing of it, via the `* 5` multiplier, is the defect.
    assert "self._cached_log_interval * 5" not in src, (
        "the degraded-model-input snapshot is gated on a logging knob again"
    )


@pytest.mark.unit
def test_save_debug_snapshots_forwards_extra_under_the_diffusion_tag() -> None:
    """`degradation_source` belongs in the artifact that shows its result.

    The snapshot shows a zero-filled tensor; which tensor was zero-filled — the
    single-repetition measurement or the NEX-averaged target (#536) — is not
    recoverable from the pixels, so it rides `extra` rather than forcing a
    cross-reference to `resolved_config.json`.
    """
    mock = MagicMock()
    tensors = {"noisy_kspace": torch.zeros(1, 2, 4, 4), "mask": torch.ones(1, 1, 4, 4)}

    DiffusionTrainingStrategy._save_debug_snapshots(
        mock, tensors, 1, extra={"degradation_source": "input"}
    )

    _, kwargs = mock.save_debug_snapshot.call_args
    assert kwargs["tag"] == "diffusion_step"
    assert kwargs["extra"] == {"degradation_source": "input"}
    assert "noisy_kspace" in kwargs["in_kspace_keys"]
    assert "mask" not in kwargs["in_kspace_keys"]


@pytest.mark.unit
def test_extra_is_optional_so_other_callers_are_unaffected() -> None:
    mock = MagicMock()

    DiffusionTrainingStrategy._save_debug_snapshots(
        mock, {"noisy_kspace": torch.zeros(1, 2, 4, 4)}, 1
    )

    _, kwargs = mock.save_debug_snapshot.call_args
    assert kwargs["extra"] is None


# ---------------------------------------------------------------------------
# Validation no longer clamps the prediction to the target's range (2026-08-18)
#
# experiment_11_attention_none wrote uniformly WHITE fake renders at every
# cascade level, reproduced byte-for-byte on two independent clusters. Every
# saved sample was bit-exactly constant at 3.8493783473968506, which is
# float32(3.49943470954895) * float32(1.1) -- the batch target's peak
# magnitude times the clamp's 1.1 headroom, i.e. the ceiling and nothing but
# the ceiling.
#
# The clamp bounded the prediction by a statistic of the GROUND TRUTH, so a
# model running hot was rewritten into a plausible-looking flat field instead
# of being reported, and PSNR/SSIM were scored against that rewrite. These pin
# the retraction and the measurement that replaced it.
# ---------------------------------------------------------------------------

_EXPERIMENT_11_TARGET_PEAK = 3.49943470954895
_EXPERIMENT_11_WHITE_CONSTANT = 3.8493783473968506


@pytest.mark.unit
def test_the_observed_white_constant_is_the_retracted_clamp_ceiling() -> None:
    """Anchor the regression to the number that was actually on disk.

    If this drifts, the tests below are pinning something other than the
    failure they were written for.
    """
    ceiling = (torch.tensor(_EXPERIMENT_11_TARGET_PEAK) * 1.1).item()

    assert ceiling == _EXPERIMENT_11_WHITE_CONSTANT


@pytest.mark.unit
def test_validation_does_not_bound_the_prediction_by_the_target() -> None:
    """The clamp is gone from the validation metrics path.

    A source-level pin, because the alternative -- driving the whole of
    ``_compute_validation_metrics`` -- needs a live model, scheduler and
    metrics service. What must never come back is *any* rewrite of the
    prediction whose bound is derived from the target.
    """
    import inspect

    src = inspect.getsource(DiffusionTrainingStrategy._compute_validation_metrics)
    # Comment lines are excluded on purpose: the retraction documents the
    # removed expression verbatim so the next reader knows what NOT to add back.
    code = [line for line in src.splitlines() if not line.strip().startswith("#")]

    assert not any("abs().max() * 1.1" in line for line in code)
    # No clamp of the prediction against a target-derived bound, in any spelling.
    for line in code:
        assert not ("clamp" in line and "target" in line), line


@pytest.mark.unit
def test_a_hot_prediction_is_measured_not_rewritten() -> None:
    """The replacement reports the scale gap instead of erasing it."""
    target = torch.full((2, 1, 8, 8), _EXPERIMENT_11_TARGET_PEAK)
    # Every element above the target's peak -- the uniformly-white case.
    pred = torch.full((2, 1, 8, 8), 12.0)

    scale = DiffusionTrainingStrategy._measure_prediction_scale(pred, target)

    assert scale["pred_above_target_fraction"] == 1.0
    assert scale["pred_abs_max"] == pytest.approx(12.0)
    assert scale["target_abs_max"] == pytest.approx(_EXPERIMENT_11_TARGET_PEAK)
    assert scale["pred_target_scale_ratio"] == pytest.approx(12.0 / _EXPERIMENT_11_TARGET_PEAK)
    # And the measurement is loud enough to trip the warning threshold.
    assert (
        scale["pred_above_target_fraction"] >= DiffusionTrainingStrategy._PRED_SCALE_WARN_FRACTION
    )


@pytest.mark.unit
def test_a_well_scaled_prediction_measures_quiet() -> None:
    """A prediction inside the target's range must not trip the warning."""
    target = torch.linspace(0.0, 4.0, 64).reshape(1, 1, 8, 8)
    pred = target * 0.9

    scale = DiffusionTrainingStrategy._measure_prediction_scale(pred, target)

    assert scale["pred_above_target_fraction"] == 0.0
    assert scale["pred_target_scale_ratio"] == pytest.approx(0.9)
    assert scale["pred_above_target_fraction"] < DiffusionTrainingStrategy._PRED_SCALE_WARN_FRACTION


@pytest.mark.unit
def test_a_uniformly_attenuated_prediction_is_invisible_to_the_peak_statistics() -> None:
    """PLANTED: the experiment_11_attention_none failure shape, R=32.

    Both incumbent numbers are driven by the PEAK, so neither can see a deficit
    spread over the whole image. Measured on that arm's own saved validation case
    (``report_cases/case_2.npz``, 2026-09-16): a reconstruction carrying 0.295 of
    the target's energy reported ``pred_target_scale_ratio`` 1.075 and
    ``pred_above_target_fraction`` ~0 -- both squarely in band, on a 70000-iteration
    run whose images are visibly zero-filled.

    This test pins the MISS as well as the catch. Without the first two asserts a
    later simplification could drop the energy statistics and leave the gate blind
    again with nothing turning red.
    """
    torch.manual_seed(0)
    target = torch.rand(2, 1, 32, 32) * _EXPERIMENT_11_TARGET_PEAK
    # A low-passed copy at ~55% amplitude, with one hot pixel so the peak agrees:
    # exactly the shape hard DC produces when the null band is under-filled.
    pred = target * 0.55
    pred[0, 0, 0, 0] = target.max() * 1.05

    scale = DiffusionTrainingStrategy._measure_prediction_scale(pred, target)

    # The miss: the incumbents read healthy.
    assert scale["pred_target_scale_ratio"] == pytest.approx(1.05, abs=1e-3)
    assert scale["pred_above_target_fraction"] < DiffusionTrainingStrategy._PRED_SCALE_WARN_FRACTION

    # The catch: energy and the least-squares gain both name the deficit, and
    # the gain names its SIZE -- the factor a single scalar would recover.
    assert scale["pred_target_energy_ratio"] < 0.35
    assert scale["pred_target_optimal_gain"] > 1.7


@pytest.mark.unit
def test_an_honestly_scaled_prediction_reads_one_on_both_energy_statistics() -> None:
    """The anchor: identical tensors give ratio 1.0 and gain 1.0."""
    target = torch.linspace(0.5, 4.0, 64).reshape(1, 1, 8, 8)

    scale = DiffusionTrainingStrategy._measure_prediction_scale(target.clone(), target)

    assert scale["pred_target_energy_ratio"] == pytest.approx(1.0, abs=1e-5)
    assert scale["pred_target_optimal_gain"] == pytest.approx(1.0, abs=1e-5)

    # A globally scaled prediction: energy goes as the square, the gain is the
    # exact factor that undoes it. This is what makes the gain readable as
    # "the reconstruction is under-scaled by 2x" rather than as an index.
    half = DiffusionTrainingStrategy._measure_prediction_scale(target * 0.5, target)
    assert half["pred_target_energy_ratio"] == pytest.approx(0.25, abs=1e-5)
    assert half["pred_target_optimal_gain"] == pytest.approx(2.0, abs=1e-5)


@pytest.mark.unit
def test_prediction_scale_is_measured_on_magnitude_for_complex_tensors() -> None:
    """Complex predictions are compared in the domain the render path sees.

    The old clamp skipped complex tensors entirely, so complex validation
    paths had no scale readout at all.
    """
    target = torch.complex(torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    pred = torch.complex(torch.zeros(1, 1, 4, 4), torch.full((1, 1, 4, 4), 3.0))

    scale = DiffusionTrainingStrategy._measure_prediction_scale(pred, target)

    assert scale["pred_abs_max"] == pytest.approx(3.0)
    assert scale["target_abs_max"] == pytest.approx(1.0)
    assert scale["pred_above_target_fraction"] == 1.0


@pytest.mark.unit
def test_a_zero_target_does_not_divide_by_zero() -> None:
    """An all-zero target batch must yield a finite ratio, not inf/NaN."""
    target = torch.zeros(1, 1, 4, 4)
    pred = torch.ones(1, 1, 4, 4)

    scale = DiffusionTrainingStrategy._measure_prediction_scale(pred, target)

    assert torch.isfinite(torch.tensor(scale["pred_target_scale_ratio"]))
    assert scale["pred_above_target_fraction"] == 1.0


@pytest.mark.unit
def test_the_validation_path_no_longer_writes_to_a_hardcoded_tmp_path() -> None:
    """Production validation must not dump tensors to /tmp on every batch.

    Left over from the March 2026 N/2-ghosting investigation (e6f967c51),
    alongside the clamp.
    """
    import inspect

    src = inspect.getsource(DiffusionTrainingStrategy._compute_validation_metrics)

    assert "/tmp/val_diag" not in src
    assert "torch.save(" not in src


@pytest.mark.unit
def test_prediction_scale_survives_a_half_precision_autocast_batch() -> None:
    """Half-precision inputs measure finite -- ``eps`` must not underflow to zero.

    Validation may sample under ``torch.amp.autocast``, so the prediction can
    arrive fp16 while the target arrives fp32 from the loader. The zero-target
    guard is ``clamp(min=1e-12)``, and ``1e-12`` is exactly ``0.0`` in fp16:
    computing the ratio in the narrow dtype divides by zero and reports
    ``inf`` for a silent target instead of a large finite number.
    """
    for pred_dtype, target_dtype in (
        (torch.float16, torch.float32),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.float32),
    ):
        pred = torch.full((1, 1, 4, 4), 3.0, dtype=pred_dtype)
        target = torch.full((1, 1, 4, 4), 1.0, dtype=target_dtype)
        measured = DiffusionTrainingStrategy._measure_prediction_scale(pred, target)
        assert measured["pred_above_target_fraction"] == 1.0, (pred_dtype, target_dtype)
        assert measured["pred_target_scale_ratio"] == pytest.approx(3.0), (
            pred_dtype,
            target_dtype,
        )

    # A fully silent fp16 target is the case that used to divide by zero.
    silent = DiffusionTrainingStrategy._measure_prediction_scale(
        torch.ones(1, 1, 4, 4, dtype=torch.float16),
        torch.zeros(1, 1, 4, 4, dtype=torch.float16),
    )
    assert math.isfinite(silent["pred_target_scale_ratio"])


# Validation render: the log-decompression must verify its declaration (#682)
#
# ``_compute_validation_metrics`` used to call ``decompress_kspace_log``
# directly, gated only on the declared ``data.processing.enable_log_scaling``.
# On a tensor that was never compressed, ``expm1`` under the
# ``DECOMPRESS_MAGNITUDE_CEILING`` clamp flattens the entire magnitude band onto
# ``expm1(30)``, leaving a constant-magnitude / varying-phase spectrum. The
# tensors it corrupts are ``log_preds``/``log_targets`` -- the single source of
# BOTH the saved validation PNGs and the validation metrics -- so the damage is
# silent in every artifact at once.
# ---------------------------------------------------------------------------


def _multicoil_kspace(peak: float, coils: int = 4, size: int = 32) -> torch.Tensor:
    """Real-stacked ``[1, 2*coils, H, W]`` k-space scaled to ``|k|max == peak``."""
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    disc = ((yy**2 + xx**2).sqrt() < 0.7).float()
    img = disc[None, None].repeat(1, coils, 1, 1).to(torch.complex64)
    ksp = fft2c(img)  # non-negotiable 2: never raw torch.fft.*
    stacked = torch.empty(1, 2 * coils, size, size)
    stacked[:, 0::2], stacked[:, 1::2] = ksp.real, ksp.imag
    return stacked * (peak / stacked.abs().max())


def _magnitudes(stacked: torch.Tensor) -> torch.Tensor:
    """Per-coil complex magnitudes — the quantity spurious expm1 destroys."""
    return torch.complex(stacked[:, 0::2].contiguous(), stacked[:, 1::2].contiguous()).abs()


def _distinct_magnitudes(stacked: torch.Tensor) -> int:
    return int(torch.unique(_magnitudes(stacked)).numel())


def test_unguarded_expm1_would_collapse_the_magnitude_band() -> None:
    """The defect, pinned as the exact mechanism rather than an aggregate ratio.

    This is what the validation path did on every batch of the
    ``experiment_11_attention_none`` run: the arm declared
    ``enable_log_scaling: true`` while the batch arrived uncompressed at
    ``|k|max ~ 2479``, so ``mag.clamp(max=CEILING)`` flattened every magnitude
    above the ceiling onto one value before ``expm1``.

    How *much* of the band collapses depends on the spectrum — on the run's own
    tensors the prediction went from 1087 distinct magnitudes to 5, whereas this
    hard-edged phantom keeps most of its energy below the ceiling. So assert the
    invariant that holds for any input: the above-ceiling bins become
    indistinguishable from each other. Their differences carried the contrast,
    and they are gone, not merely rescaled.
    """
    ksp = _multicoil_kspace(peak=2478.8)  # the arm's own measured |k|max
    before = _magnitudes(ksp)
    over_ceiling = before > DECOMPRESS_MAGNITUDE_CEILING
    assert int(over_ceiling.sum()) > 0, "phantom must exercise the clamp"

    after = _magnitudes(decompress_kspace_log(ksp, channel_dim=1))

    def _rel_spread(t: torch.Tensor) -> float:
        return float((t.max() - t.min()) / t.max())

    # The above-ceiling bins spanned nearly their whole range before...
    assert _rel_spread(before[over_ceiling]) > 0.9
    # ...and afterwards are one value, to float32 precision. (Not bit-identical:
    # the rescale reconstructs as ``z * (new_mag / old_mag)``, so rounding leaves
    # a handful of adjacent floats — which is still total information loss.)
    assert _rel_spread(after[over_ceiling]) < 1e-5
    assert float(after[over_ceiling].max()) == pytest.approx(
        math.expm1(DECOMPRESS_MAGNITUDE_CEILING), rel=1e-3
    )


def test_validation_decompress_skips_expm1_on_never_compressed_kspace() -> None:
    """Declared log-scaled, but the tensor says otherwise → leave it alone."""
    ksp = _multicoil_kspace(peak=2478.8)
    before = _distinct_magnitudes(ksp)

    out = decompress_for_view(ksp, log_scaled=True, channel_dim=1)

    assert torch.equal(out, ksp)  # returned as-is, not expm1'd
    assert _distinct_magnitudes(out) == before


def test_validation_decompress_still_applies_expm1_when_genuinely_compressed() -> None:
    """The guard must not disarm the real inverse.

    A ``log1p``-compressed magnitude cannot exceed ~6 for physical data, so this
    tensor is below the ceiling and ``expm1`` is the correct operation.
    """
    ksp = _multicoil_kspace(peak=6.0)

    out = decompress_for_view(ksp, log_scaled=True, channel_dim=1)

    assert float(out.abs().max()) > float(ksp.abs().max())  # expm1 strictly expands
    assert not torch.equal(out, ksp)


def test_validation_decompress_is_a_noop_when_not_declared() -> None:
    """``log_scaled=False`` must not touch the tensor at any magnitude."""
    ksp = _multicoil_kspace(peak=6.0)

    assert torch.equal(decompress_for_view(ksp, log_scaled=False, channel_dim=1), ksp)


def test_validation_metrics_routes_decompression_through_the_verifying_helper() -> None:
    """Pin the bypass shut.

    ``_compute_validation_metrics`` must not reach ``decompress_kspace_log``
    directly again — that call skips the declared-vs-actual check entirely.
    Asserted over the AST, not a substring search, so the prose in this method's
    own comments cannot satisfy or trip the check
    (cf. census-by-AST: a call-site regex under-counts precisely where the code
    is tidiest).
    """
    src = textwrap.dedent(inspect.getsource(DiffusionTrainingStrategy._compute_validation_metrics))
    called = {
        node.func.id
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "decompress_kspace_log" not in called
    # The verifying entry for tensors scored against the target: it runs
    # decompress_for_view's check once, on the target, for all of them.
    assert "decompress_scored_against" in called


def _scored_against_the_target(tree: ast.AST) -> set[str]:
    """Names passed into ``decompress_scored_against`` (the reference and each ``others`` value)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "decompress_scored_against"
        ):
            continue
        if node.args and isinstance(node.args[0], ast.Name):
            names.add(node.args[0].id)
        for arg in node.args[1:2]:
            if isinstance(arg, ast.Dict):
                names.update(v.id for v in arg.values if isinstance(v, ast.Name))
    return names


# ---------------------------------------------------------------------------
# _compute_validation_metrics: denom_scale must be sized from the prediction
#
# Regression anchor (2026-08-19): a 40-iteration cluster relaunch of
# ``experiment_11_attention_none`` trained fine, then died at the first
# validation step::
#
#     diffusion.py:4758  hr_fakes_for_metrics = hr_fakes * denom_scale
#     RuntimeError: The size of tensor a (36) must match the size of
#                   tensor b (2) at non-singleton dimension 0
#
# ``36 = 2 subjects x 18 slices``. This method sizes ``denom_scale`` correctly
# from ``hr_fakes.size(0)`` and then replaced it with the per-SUBJECT published
# field via ``view(-1, 1, 1, 1)``, adopting the producer's length unchecked.
#
# This site's ``read_batch_field`` is an INDEPENDENT read -- it fires whenever
# ``scale_factor`` arrives ``None`` -- so fixing only ``_prepare_validation_data``
# would leave this path live.
# ---------------------------------------------------------------------------


def _cold_diffusion_mock() -> MagicMock:
    """A ``self`` exposing only what the scale block ahead of :4758 reads."""
    from spectramr.config.schemas.data import DataProcessingConfigSchema

    mock = MagicMock()
    mock.config = SimpleNamespace(
        data=SimpleNamespace(
            processing=DataProcessingConfigSchema(
                enable_kspace_normalization=True, enable_log_scaling=False
            )
        ),
    )
    mock._is_cold_diffusion = MagicMock(return_value=True)
    return mock


def _call_with_flattened_batch(mock, published_scale, batch=36):
    """Drive the method with a per-slice prediction and a per-subject scale.

    Returns whatever exception the call ultimately raised (or ``None``). The
    mock cannot satisfy metric computation past the multiply -- on this tree it
    trips at ``self.config.validation`` (:4867) -- so the *identity* of the
    error is the observable, not a successful return. Reachability of the seam
    is asserted separately by the spy, so a call that dies early cannot pass
    vacuously.
    """
    hr = torch.randn(batch, 1, 8, 8)
    tgt = torch.randn(batch, 1, 8, 8)
    inp = torch.randn(batch, 1, 8, 8)
    batch_data = {"kspace_scale": published_scale}
    try:
        DiffusionTrainingStrategy._compute_validation_metrics(
            mock, hr, tgt, inp, torch.zeros(batch), batch_data, None
        )
    except Exception as exc:  # the error identity IS the assertion, see docstring
        return exc
    return None


def _spy_on_alignment(monkeypatch):
    """Record every ``align_scale_to_batch`` call the strategy makes."""
    from spectramr.data.batch_types import align_scale_to_batch as real
    from spectramr.infrastructure.training.strategies import diffusion as diffusion_mod

    calls: list[dict] = []

    def spy(scale, batch_size, **kwargs):
        aligned = real(scale, batch_size, **kwargs)
        calls.append(
            {
                "batch_size": batch_size,
                "published": tuple(torch.as_tensor(scale).shape),
                "aligned": aligned,
            }
        )
        return aligned

    monkeypatch.setattr(diffusion_mod, "align_scale_to_batch", spy)
    return calls


def test_denom_scale_is_sized_from_the_prediction_not_the_published_field(
    monkeypatch,
) -> None:
    """The batch size must be read off ``hr_fakes``, never off the batch field.

    This is the defect stated directly: the consumer knows how many entries it
    needs (36) and the producer published a different number (2). Taking the
    producer's is what reached the multiply.
    """
    calls = _spy_on_alignment(monkeypatch)
    _call_with_flattened_batch(_cold_diffusion_mock(), torch.tensor([224.36, 198.15]), batch=36)

    assert calls, "the scale-alignment seam was never reached"
    assert calls[0]["batch_size"] == 36
    assert calls[0]["published"] == (2,)


def test_the_multiply_that_crashed_the_cluster_no_longer_collides(
    monkeypatch,
) -> None:
    """The exact ``36 vs 2`` RuntimeError must not recur."""
    calls = _spy_on_alignment(monkeypatch)
    error = _call_with_flattened_batch(
        _cold_diffusion_mock(), torch.tensor([224.36, 198.15]), batch=36
    )

    assert calls, "the scale-alignment seam was never reached"
    assert "must match the size" not in str(error or ""), f"the 36-vs-2 collision is back: {error}"


def test_the_aligned_scale_is_subject_major_at_this_site_too(monkeypatch) -> None:
    """Values, not shapes -- ``repeat`` has the same shape and is wrong here too.

    The flatten is ``permute(0, 4, 1, 2, 3).reshape(B * D, ...)``, so slice
    ``b * D + d`` belongs to subject ``b``: all of subject 0's slices, then
    subject 1's. Interleaving would grade subject 1 in subject 0's units with no
    error raised.
    """
    calls = _spy_on_alignment(monkeypatch)
    _call_with_flattened_batch(_cold_diffusion_mock(), torch.tensor([10.0, 20.0]), batch=6)

    assert calls, "the scale-alignment seam was never reached"
    assert calls[0]["aligned"].flatten().tolist() == [
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
    ]


def test_an_unalignable_published_scale_raises_with_the_field_named() -> None:
    """A non-dividing length has no benign reading (non-negotiable 3).

    Before this change it surfaced as a bare ``RuntimeError`` at a multiply,
    naming neither the field nor the producer.
    """
    error = _call_with_flattened_batch(_cold_diffusion_mock(), torch.ones(5), batch=36)

    assert isinstance(error, ValueError), f"expected ValueError, got {error!r}"
    assert "kspace_scale" in str(error)


# ── #1298: one rule for which snapshot key is the model input ───────────────


def test_the_model_input_key_follows_whether_smaps_were_concatenated() -> None:
    """The label must track what was recorded, not a constant.

    It read `"noisy_kspace"` unconditionally, which is true only for an arm
    without smaps conditioning. With it -- the default for
    `kspace_cold_diffusion`, and what the attention arms run -- the network is
    fed the 16-channel concat and `noisy_kspace` is merely its first half.
    """
    from spectramr.infrastructure.training.strategies.diffusion import (
        cold_model_input_key,
    )

    assert cold_model_input_key({"noisy_kspace": 1, "target": 2}) == "noisy_kspace"
    assert cold_model_input_key({"noisy_kspace": 1, "model_input": 2, "smaps": 3}) == "model_input"


def test_both_cold_emitters_answer_through_the_same_rule() -> None:
    """The declaring path and the direct `diffusion_step` emitter must agree.

    They are in different methods 2000 lines apart. Written out twice, one of
    them gets a later fix the other misses -- which is #697, and #1298 is what a
    snapshot disagreeing with its own label costs.
    """
    import inspect

    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    for method in (
        DiffusionTrainingStrategy._prepare_diffusion_inputs,
        DiffusionTrainingStrategy._save_debug_snapshots,
    ):
        src = inspect.getsource(method)
        assert "cold_model_input_key(" in src, (
            f"{method.__name__} answers 'which key is the model input' without "
            "going through the shared rule"
        )
        assert '"model_input" if "model_input" in' not in src, (
            f"{method.__name__} still carries a hand-written copy of the rule"
        )


# ---------------------------------------------------------------------------
# S-maps are FFT'd before they are concatenated onto a k-space input (#1297)
# ---------------------------------------------------------------------------
#
# ``FourierBridgeNetwork.forward`` applies ONE domain transform to the whole
# concatenated stack, so image-domain maps riding next to k-space are mistreated
# whichever way ``force_pure_kspace`` is set. The strategy builds that stack in
# TWO places -- training (``_prepare_diffusion_inputs``) and validation
# (``_generate_validation_prediction``) -- and they must agree, or the model is
# validated on a stack it was never trained on (the shape of #1295).

_SMAPS_HELPER = "prepare_smaps_for_kspace_conditioning"


def _concat_arg_names(func) -> list[list[str]]:
    """Names passed to each ``torch.cat([...])`` in ``func``, as an AST census.

    A call-site regex under-counts exactly where the code is tidiest, so this
    walks the tree instead.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    out: list[list[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if not (isinstance(target, ast.Attribute) and target.attr == "cat"):
            continue
        if not node.args or not isinstance(node.args[0], (ast.List, ast.Tuple)):
            continue
        out.append([e.id for e in node.args[0].elts if isinstance(e, ast.Name)])
    return out


def _calls_named(func, name: str) -> int:
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "method_name",
    ["_prepare_diffusion_inputs", "_generate_validation_prediction"],
)
def test_smaps_are_prepared_before_the_kspace_concat(method_name: str) -> None:
    """Both stack-building sites transform the maps first, and agree.

    Pinned as a pair: a fix applied to training alone would reintroduce a
    train/val skew, which is the failure mode that motivated this at all.
    """
    method = getattr(DiffusionTrainingStrategy, method_name)
    assert _calls_named(method, _SMAPS_HELPER) == 1, method_name
    concats = [names for names in _concat_arg_names(method) if "smaps_k" in names]
    assert concats, f"{method_name} does not concatenate the prepared maps"
    for names in concats:
        assert "smaps" not in names, f"{method_name} still concatenates the raw image-domain maps"


@pytest.mark.unit
def test_current_smaps_is_never_rebound_to_the_kspace_form() -> None:
    """``_current_smaps`` must stay IMAGE-domain -- three consumers depend on it.

    The SENSE projection (``sense_projector``), the KAN FiLM stash and
    ``gen_kwargs["smaps"]`` all read this attribute and all need image space.
    The fix therefore rebinds a LOCAL name only; assigning the prepared tensor
    to the attribute would silently break the one physically-correct use of the
    maps in the whole pipeline.
    """
    for method_name in (
        "_prepare_diffusion_inputs",
        "_generate_validation_prediction",
    ):
        src = textwrap.dedent(inspect.getsource(getattr(DiffusionTrainingStrategy, method_name)))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                is_current_smaps = (
                    isinstance(tgt, ast.Attribute)
                    and tgt.attr == "_current_smaps"
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                )
                if is_current_smaps:
                    assert isinstance(node.value, ast.Name), (
                        f"{method_name}: _current_smaps assigned a non-trivial "
                        "expression -- check it is still the image-domain maps"
                    )
                    assert node.value.id == "smaps", (
                        f"{method_name}: _current_smaps must stay image-domain, "
                        f"got '{node.value.id}'"
                    )


@pytest.mark.unit
def test_snapshot_declares_the_concatenated_smaps_as_kspace() -> None:
    """Non-negotiable 14: the artifact must say what domain it stored.

    Before the fix the concatenated half really was image-domain and was
    deliberately excluded from ``in_kspace_keys`` (VIS-1). It is k-space now, so
    the previewer has to IFFT it -- and the comment that justified excluding it
    has to go with it, or the next reader trusts a stale claim.
    """
    src = textwrap.dedent(inspect.getsource(DiffusionTrainingStrategy._prepare_diffusion_inputs))
    assert '"noisy_kspace", "target", "smaps"' in src
    assert "smaps are image-domain despite their real-stacked" not in src
    assert "smaps_conditioning" in src
    assert "SMAP_KSPACE_PEAK_RATIO" in src


# ── Multi-step cold validation must sample at the trained width (#1326) ───────
#
# ``_generate_validation_prediction``'s cold fork hands ``masked_input`` (pure
# measured k-space) to ``gen.sample()``. It used to hand over nothing else, so
# the reverse loop ran the model on a bare ``in_channels`` stack while training
# fed ``2 * in_channels``. ``FourierBridgeNetwork`` absorbed the difference by
# rebuilding an untrained 1x1 ChannelAdapter, so validation reconstructed
# through random weights — experiment_11_attention_none's R8x/R32x fakes came
# out anticorrelated with their targets while R2x looked fine, because at R2x
# hard DC pins half of k-space over the garbage.


class _RecordingGen:
    """Stands in for the generator: records what ``sample()`` was given."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def sample(self, *, measurement, mask, inference_timesteps, **kwargs):
        self.calls.append(
            {
                "measurement": measurement,
                "mask": mask,
                "steps": inference_timesteps,
                "smaps": kwargs.get("smaps"),
                "reference": kwargs.get("reference"),
                "contrast_idx": kwargs.get("contrast_idx"),
            }
        )
        return torch.zeros_like(measurement)


def _chunking_strategy(chunk_size: int):
    """Minimal ``self`` for an unbound ``_sample_multistep_chunked`` call."""
    return SimpleNamespace(
        config=SimpleNamespace(
            validation=SimpleNamespace(loader=SimpleNamespace(chunk_size=chunk_size))
        )
    )


@pytest.mark.unit
def test_multistep_sampling_hands_the_smaps_to_the_generator() -> None:
    """The unchunked path forwards the maps. Without them the generator raises
    rather than silently reconstructing through an untrained projection."""
    gen = _RecordingGen()
    meas = torch.randn(2, 4, 8, 8)
    smaps = torch.randn(2, 2, 8, 8, dtype=torch.complex64)
    DiffusionTrainingStrategy._sample_multistep_chunked(
        _chunking_strategy(8), gen, meas, None, 3, smaps=smaps
    )
    assert len(gen.calls) == 1
    assert gen.calls[0]["smaps"] is smaps
    # The maps must NOT be folded into the measurement: sample() runs its own
    # physics and needs pure measured k-space.
    assert gen.calls[0]["measurement"].shape[1] == meas.shape[1]


@pytest.mark.unit
def test_multistep_chunking_splits_the_smaps_with_the_measurement() -> None:
    """Chunk boundaries must line up, or one subject's coil profile is applied
    to another subject's k-space — a silent, per-element wrong answer."""
    gen = _RecordingGen()
    meas = torch.randn(4, 4, 8, 8)
    smaps = torch.randn(4, 2, 8, 8, dtype=torch.complex64)
    DiffusionTrainingStrategy._sample_multistep_chunked(
        _chunking_strategy(2), gen, meas, None, 3, smaps=smaps
    )
    assert len(gen.calls) == 2
    for i, call in enumerate(gen.calls):
        assert call["smaps"] is not None
        assert call["smaps"].shape[0] == call["measurement"].shape[0] == 2
        assert torch.equal(call["smaps"], smaps[2 * i : 2 * i + 2])
        assert torch.equal(call["measurement"], meas[2 * i : 2 * i + 2])


@pytest.mark.unit
def test_multistep_chunking_splits_the_reference_with_the_measurement() -> None:
    """The T1 reference is per row: a misaligned chunk would condition one
    subject's target on another subject's T1 (#2452)."""
    gen = _RecordingGen()
    meas = torch.randn(5, 8, 8, 8)
    ref = torch.randn(5, 8, 8, 8)
    contrast = torch.arange(5)
    DiffusionTrainingStrategy._sample_multistep_chunked(
        _chunking_strategy(2), gen, meas, None, 3, contrast_idx=contrast, reference=ref
    )
    assert [c["measurement"].shape[0] for c in gen.calls] == [2, 2, 1]
    for i, call in enumerate(gen.calls):
        rows = slice(2 * i, 2 * i + call["measurement"].shape[0])
        assert torch.equal(call["reference"], ref[rows])
        assert torch.equal(call["contrast_idx"], contrast[rows])


@pytest.mark.unit
def test_multistep_sampling_passes_the_reference_unchunked() -> None:
    gen = _RecordingGen()
    ref = torch.randn(2, 8, 8, 8)
    DiffusionTrainingStrategy._sample_multistep_chunked(
        _chunking_strategy(8), gen, torch.randn(2, 8, 8, 8), None, 3, reference=ref
    )
    assert gen.calls[0]["reference"] is ref


@pytest.mark.unit
def test_multistep_sampling_tolerates_absent_smaps() -> None:
    """A paradigm with no maps still samples; the generator decides whether it
    needs them (``expects_smaps_concat``), not this method."""
    gen = _RecordingGen()
    meas = torch.randn(3, 4, 8, 8)
    DiffusionTrainingStrategy._sample_multistep_chunked(
        _chunking_strategy(2), gen, meas, None, 3, smaps=None
    )
    assert [c["smaps"] for c in gen.calls] == [None, None]


@pytest.mark.unit
def test_the_training_concat_obeys_the_generators_sizing_predicate() -> None:
    """One owner (CLAUDE.md #17): the strategy must ASK the resolver whether the
    backbone was sized for a doubled stack, never re-derive the rule.

    It used to concatenate whenever maps existed, knowing nothing about the
    backbone's width, so the six internal-DC arms (diff_varnet x4,
    diff_varnet_kan x2) fed ``2 * C`` into a ``C``-wide backbone on every
    training step for their whole run.
    """
    method = DiffusionTrainingStrategy._prepare_diffusion_inputs
    assert _calls_named(method, "model_expects_smaps_concat") == 1, (
        "_prepare_diffusion_inputs does not ask the one resolver whether the "
        "backbone was sized for a doubled stack; the concat and the width are "
        "then two owners of one rule."
    )
    # ...and it must not re-derive the rule from the arm's *declaration*, which
    # stays True on the internal-DC arms whose backbone is built at 1x.
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    losers = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and n.value == "condition_with_smaps"
    ]
    assert not losers, (
        "_prepare_diffusion_inputs reads condition_with_smaps directly; that is "
        "the declaration, not the resolved contract (CLAUDE.md #17)."
    )


class TestMultistepStartTimestepPassthrough:
    """The cascade's per-rung timestep survives the chunked reverse-sampling path.

    Cascading validation evaluates R=2/8/32 against one sampler. Each rung's input
    is degraded at its OWN timestep, but ``sample()`` used to start every trajectory
    fully degraded, so every rung below the top replayed steps that could not write
    a coefficient (#535/#1388). The strategy now passes the rung's ``t_used`` down.

    ``_sample_multistep_chunked`` reaches ``gen.sample`` through TWO call sites -- a
    whole-batch shortcut and a per-chunk loop -- and micro-chunking is exactly what
    validation runs under (``chunk_size=1`` on experiment_11), so the loop site is
    the one production uses. Both are pinned.
    """

    class _RecordingGen:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def sample(self, measurement, mask, inference_timesteps, smaps, start_timestep):
            self.calls.append({"n": measurement.shape[0], "start_timestep": start_timestep})
            return measurement

    @staticmethod
    def _strategy(chunk_size: int) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(
                validation=SimpleNamespace(loader=SimpleNamespace(chunk_size=chunk_size))
            )
        )

    def _run(self, chunk_size: int, batch: int, start_timestep: int | None):
        gen = self._RecordingGen()
        DiffusionTrainingStrategy._sample_multistep_chunked(
            self._strategy(chunk_size),
            gen,
            torch.zeros(batch, 2, 8, 8),
            torch.ones(batch, 1, 8, 8),
            5,
            smaps=None,
            start_timestep=start_timestep,
        )
        return gen.calls

    def test_forwarded_on_the_whole_batch_shortcut(self) -> None:
        """batch <= chunk takes the early return, which is a separate call site."""
        calls = self._run(chunk_size=4, batch=2, start_timestep=14)

        assert [c["start_timestep"] for c in calls] == [14]

    def test_forwarded_to_every_chunk(self) -> None:
        """The loop site: missing it would leave production (chunk_size=1) unfixed.

        A partial fix here is invisible -- the run still reconstructs, just at the
        old cost -- so the assertion is per-chunk rather than on the first call.
        """
        calls = self._run(chunk_size=1, batch=3, start_timestep=14)

        assert [c["n"] for c in calls] == [1, 1, 1]
        assert [c["start_timestep"] for c in calls] == [14, 14, 14]

    def test_none_is_propagated_unchanged(self) -> None:
        """Non-cascading callers keep the legacy fully-degraded head."""
        calls = self._run(chunk_size=1, batch=2, start_timestep=None)

        assert [c["start_timestep"] for c in calls] == [None, None]

    def test_validation_prediction_accepts_the_head_keyword_only(self) -> None:
        """Keyword-only with a ``None`` default, so the six other call sites are unmoved.

        Positional would silently bind to whatever a caller passed sixth.
        """
        parameters = inspect.signature(
            DiffusionTrainingStrategy._generate_validation_prediction
        ).parameters

        assert parameters["start_timestep"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["start_timestep"].default is None

    def test_cascade_passes_the_rungs_own_timestep(self) -> None:
        """``t_used`` -- the Python scalar, never read back off the timestep tensor.

        Reading it off the tensor would cost a GPU sync per rung per batch inside
        the validation loop (non-negotiable 9); the scalar is already in hand.
        """
        source = textwrap.dedent(
            inspect.getsource(DiffusionTrainingStrategy._run_cascading_validation)
            if hasattr(DiffusionTrainingStrategy, "_run_cascading_validation")
            else inspect.getsource(DiffusionTrainingStrategy)
        )
        tree = ast.parse(source)

        forwarded = {
            keyword.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_generate_validation_prediction"
            for keyword in node.keywords
            if keyword.arg == "start_timestep" and isinstance(keyword.value, ast.Name)
        }

        assert forwarded == {"t_used"}


class TestFiveDimensionalCapabilityProbe:
    """The 5D-capability probe must ask about 5D, not about repetition fusion.

    Both sites decided "may I keep this batch 5D?" with
    ``hasattr(gen, "rep_fusion") or hasattr(gen.backbone, "rep_fusion")`` --
    reaching for one invariant to answer about another (non-negotiable 17), and
    landing on a predicate that was a constant ``True`` because ``rep_fusion``
    was built on every instance (#1173).
    """

    def test_the_conflated_hasattr_predicate_is_gone(self) -> None:
        import ast
        import inspect

        from spectramr.infrastructure.training.strategies import diffusion as mod

        tree = ast.parse(inspect.getsource(mod))
        offenders = [
            ast.unparse(n)
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "hasattr"
            and len(n.args) == 2
            and isinstance(n.args[1], ast.Constant)
            and n.args[1].value == "rep_fusion"
        ]
        assert not offenders, (
            f"a 5D-capability decision is again being made from a rep_fusion hasattr: {offenders}"
        )

    def test_both_sites_ask_the_generator_for_5d_support(self) -> None:
        """PLANT: deleting the predicate without replacing it would satisfy the
        test above. Pin that the honest probe is actually the one in use, at
        both sites."""
        import inspect

        from spectramr.infrastructure.training.strategies import diffusion as mod

        src = inspect.getsource(mod)
        assert src.count('getattr(gen, "supports_5d_input", False)') == 2

    def test_the_default_is_false_for_a_generator_without_the_property(self) -> None:
        """Absent is reported, not inferred: a generator that does not publish
        the property has not declared 5D support, so the caller flattens."""
        from types import SimpleNamespace

        assert bool(getattr(SimpleNamespace(), "supports_5d_input", False)) is False


# ---------------------------------------------------------------------------
# The fully-sampled rung's only gradient path (issue #535, train_identity_rung)
# ---------------------------------------------------------------------------
#
# When an arm opts into `undersampling.train_identity_rung`, t=0 is drawn with
# an all-ones mask. Hard DC then replaces the network's proposal at every
# acquired bin -- which at t=0 is every bin -- so every post-DC loss term is a
# constant with zero gradient. Measured on the real generator at t=0:
#
#     grad_norm, post-DC term only            = 0.000000e+00
#     grad_norm, total, lambda_pre_dc = 0.3   = 3.845772e-02
#     grad_norm, total, lambda_pre_dc = 0.0   = 0.000000e+00
#
# The rung learns, and it learns ONLY through the pre-DC term -- which reaches
# it via `_unsampled_weight` returning None and the caller falling back to a
# uniform L1. These pin that chain. Both would still pass if the rung had
# silently stopped learning were they written as "does not crash", so each
# asserts the VALUE, not the absence of an exception.


def test_unsampled_weight_returns_none_when_fully_sampled() -> None:
    """CONTRACT: None routes the caller to the uniform L1 fallback.

    A "tidied" implementation returning a zeroed weight tensor instead would
    make the t=0 rung's loss identically zero -- the loss still computes,
    training still runs, and nothing goes red.
    """
    ref = torch.randn(2, 8, 16, 16)
    mask = torch.ones(2, 1, 16, 16)
    assert DiffusionTrainingStrategy._unsampled_weight(mask, ref) is None, (
        "an all-ones mask must yield None, not a zeroed weight"
    )


def test_unsampled_weight_is_not_none_when_bins_are_missing() -> None:
    """The discriminating half: the None above is about FULL sampling.

    Without this, an implementation that returned None unconditionally would
    satisfy the test above while destroying the term on every other rung.
    """
    ref = torch.randn(2, 8, 16, 16)
    mask = torch.ones(2, 1, 16, 16)
    mask[:, :, 8:, :] = 0.0
    weight = DiffusionTrainingStrategy._unsampled_weight(mask, ref)
    assert weight is not None
    assert float(weight.sum()) > 0.0


def test_pre_dc_fidelity_falls_back_to_uniform_l1_at_full_sampling() -> None:
    """The t=0 rung's entire learning signal, asserted as a VALUE.

    ``total_loss + lam * diff.mean()`` is the uniform fallback. Asserting the
    exact arithmetic is what separates "the fallback fired" from "the term was
    silently dropped and total_loss came back unchanged".
    """
    strategy = MagicMock(spec=DiffusionTrainingStrategy)
    strategy._loss_dict_reuse = {}
    strategy.config = SimpleNamespace(
        losses=SimpleNamespace(reconstruction=SimpleNamespace(lambda_pre_dc_kspace=0.3))
    )
    strategy._unsampled_weight = DiffusionTrainingStrategy._unsampled_weight

    torch.manual_seed(0)
    post_dc = torch.randn(1, 8, 16, 16)
    pre_dc = torch.randn(1, 8, 16, 16)
    target = torch.randn(1, 8, 16, 16)
    mask = torch.ones(1, 1, 16, 16)
    total_in = torch.tensor(1.25)

    out = DiffusionTrainingStrategy._add_pre_dc_fidelity(
        strategy, total_in, (post_dc, pre_dc), target, mask
    )
    expected = total_in + 0.3 * torch.abs(pre_dc - target).mean()
    assert torch.allclose(out, expected), (float(out), float(expected))
    assert not torch.allclose(out, total_in), "the pre-DC term was dropped, not applied"
    assert "pre_dc_kspace_l1" in strategy._loss_dict_reuse


def test_fully_sampled_rung_receives_gradient_only_via_the_pre_dc_term() -> None:
    """The load-bearing observation, by GRADIENT rather than by arithmetic.

    The test above pins the loss VALUE; this pins that the value is actually
    connected to the network. They are different claims: a term computed from a
    detached tensor produces exactly the same number and teaches the rung
    nothing. The pair (a) lam=0.3 -> grad > 0 and (b) lam=0.0 -> grad == 0
    proves both that the rung learns and that the pre-DC term is the ONLY route
    by which it does.
    """
    torch.manual_seed(0)
    target = torch.randn(1, 8, 16, 16)
    mask = torch.ones(1, 1, 16, 16)

    grads: dict[float, float] = {}
    for lam in (0.3, 0.0):
        strategy = MagicMock(spec=DiffusionTrainingStrategy)
        strategy._loss_dict_reuse = {}
        strategy.config = SimpleNamespace(
            losses=SimpleNamespace(reconstruction=SimpleNamespace(lambda_pre_dc_kspace=lam))
        )
        strategy._unsampled_weight = DiffusionTrainingStrategy._unsampled_weight

        pre_dc = torch.randn(1, 8, 16, 16, requires_grad=True)
        out = DiffusionTrainingStrategy._add_pre_dc_fidelity(
            strategy, torch.zeros(()), (torch.randn(1, 8, 16, 16), pre_dc), target, mask
        )
        if out.requires_grad:
            out.backward()
        grads[lam] = 0.0 if pre_dc.grad is None else float(pre_dc.grad.abs().sum())

    assert grads[0.3] > 0.0, "the fully-sampled rung has no gradient path at all"
    assert grads[0.0] == 0.0, "something other than the pre-DC term is feeding the rung"


def test_hard_dc_discards_the_network_where_the_mask_is_one() -> None:
    """WHY the pre-DC term is the only path: post-DC has no gradient at t=0.

    ``HardDataConsistency`` blends as ``(1 - mask) * reconstruction + mask *
    kspace_obs``. At the R=1 rung the mask is all-ones, so the network's
    proposal is discarded everywhere and every post-DC loss term is a constant.
    This is asserted rather than reasoned about, because it is the premise the
    whole identity-rung change rests on.
    """
    from spectramr.infrastructure.physics.data_consistency import HardDataConsistency

    torch.manual_seed(0)
    dc = HardDataConsistency()
    dc.train()
    measurement = torch.randn(1, 8, 16, 16)
    target = torch.randn(1, 8, 16, 16)

    half = torch.ones(1, 1, 16, 16)
    half[:, :, 8:, :] = 0.0
    grads: dict[str, float] = {}
    for name, mask in (("all_ones", torch.ones(1, 1, 16, 16)), ("half", half)):
        net = torch.randn(1, 8, 16, 16, requires_grad=True)
        out = dc(net, measurement, mask.clone(), is_kspace_domain=True)
        torch.abs(out - target).mean().backward()
        grads[name] = float(net.grad.abs().sum())

    assert grads["all_ones"] == 0.0, "post-DC still carries gradient at full sampling"
    # The control: an undersampled mask DOES propagate, so the zero above is a
    # property of the all-ones mask and not of this harness.
    assert grads["half"] > 0.0, "the harness propagates no gradient for any mask"


def test_sample_timesteps_draws_the_fully_sampled_rung_at_floor_zero() -> None:
    """With the floor at 0, t=0 must actually be drawn.

    The floor is the sampler's LOWER bound; ``curriculum_start_timestep`` moves
    only the upper one, so no curriculum value can open this rung -- which is
    the defect ``train_identity_rung`` exists to fix. Measured on the real arm
    over its full 150000-iteration budget: 0 draws at floor 1, 10005 at floor 0.
    """
    drawn_at_zero = set()
    drawn_at_one = set()
    torch.manual_seed(0)
    for floor, sink in ((0, drawn_at_zero), (1, drawn_at_one)):
        strategy = _timestep_strategy("balanced_high_t", floor, curriculum_start=1)
        for iteration in range(200):
            sink.update(
                DiffusionTrainingStrategy.sample_timesteps(
                    strategy, 8, iteration=iteration
                ).tolist()
            )
    assert 0 in drawn_at_zero
    assert 0 not in drawn_at_one, "the floor of 1 must still exclude t=0"


# ---------------------------------------------------------------------------
# #1684 -- ``inputs`` must cross the SAME render seam as predictions/targets
# ---------------------------------------------------------------------------
#
# ``val/inputs`` and the report-case recorder's ``inputs`` array were built from
# the raw ``input_batch``: no ``decompress_for_view`` (undo ``log1p``), no
# ``denom_scale`` (undo the k-space percentile normalisation) and no domain
# flag. Measured on experiment_11_attention_none, that put the input panel
# ~65-100x off its own predictions/targets and made every saved case's
# ``inputs`` array incomparable with the two arrays beside it.
#
# The domain flag is the subtle half. ``inputs`` comes off the DATALOADER, so
# its domain is the *targets* element of ``needs_ifft_for_visualization`` --
# NOT ``is_preds_image``, which describes the MODEL OUTPUT and is recomputed
# after ``_apply_metric_transforms``, a transform ``inputs`` never undergoes.
# Reusing the predictions flag would skip a needed IFFT.


def _seam_mock(*, log_input=True, recorder=True, difference=False):
    """A logger mock with the three panel switches pinned to real booleans.

    ``_image_logger_mock`` leaves them as ``MagicMock`` (always truthy), which
    is fine for the #927 tests but cannot express "the consumer is OFF" -- the
    exact state the render gate below is about.
    """
    mock = _image_logger_mock()
    imgs = mock.config.logging.images
    imgs.log_validation = True
    imgs.log_input = log_input
    imgs.log_difference = difference
    imgs.max_per_batch = 4
    if recorder:
        rec = MagicMock()
        rec.enabled = True
        mock._report_case_recorder = rec
    else:
        mock._report_case_recorder = None
    mock._per_case_metric_sink = None
    return mock


def _panels(mock) -> dict:
    call = mock.logging_service.log_images_batch.call_args
    assert call is not None, "log_images_batch was never reached"
    return call.args[0]


class TestInputPanelCrossesTheRenderSeam:
    """#1684 -- the ``val/inputs`` render, its domain flag, and its gate."""

    @pytest.mark.unit
    def test_kspace_inputs_are_ifft_rendered_not_read_as_an_image(self) -> None:
        """A centred k-space delta images to a FLAT magnitude.

        This is the discriminating shape: read as k-space it IFFTs to a
        constant; read as an image (the pre-fix behaviour, and what reusing
        ``is_preds_image`` would still do on a k-space-target arm) it stays a
        single bright pixel on black. ``std`` separates the two by orders of
        magnitude, so no arrangement of the wrong branch can pass.
        """
        mock = _seam_mock()
        inputs = torch.zeros(1, 2, 16, 16)
        inputs[0, 0, 8, 8] = 1.0
        blank = torch.zeros(1, 2, 16, 16)

        DiffusionTrainingStrategy._log_validation_images_to_tensorboard(
            mock,
            blank,
            blank.clone(),
            inputs,
            {},
            batch_idx=0,
            is_image_domain=False,
            inputs_are_image=False,
        )

        panel = _panels(mock)["val/inputs"]
        assert panel.shape == (1, 1, 16, 16)
        assert panel.mean().item() > 0.0
        # Flat to within float noise; the un-IFFT'd reading has std ~0.06.
        assert panel.std().item() < 1e-5

    @pytest.mark.unit
    def test_image_domain_inputs_render_as_rss_and_are_not_refused(self) -> None:
        """The exp_hm_09_hld_mamba shape: 2-channel real-stacked IMAGE.

        ``kspace_to_image``'s F14 guard cannot tell this from k-space -- it
        keys on ``shape[1] % 2 == 0`` -- so routing the input through the
        ``already_image=True`` branch would raise on a mathematically correct
        rendering, and the method-wide ``except`` would turn that into "no
        validation images at all". The pin therefore asserts BOTH that the
        input panel is the channel-RSS and that the other panels survive.
        """
        mock = _seam_mock()
        inputs = torch.rand(1, 2, 16, 16) + 0.5
        blank = torch.zeros(1, 1, 16, 16)

        DiffusionTrainingStrategy._log_validation_images_to_tensorboard(
            mock,
            blank,
            blank.clone(),
            inputs,
            {},
            batch_idx=0,
            is_image_domain=True,
            inputs_are_image=True,
        )

        assert not _warned_about_already_image(mock)
        panels = _panels(mock)
        expected = torch.sqrt((inputs**2).sum(dim=1, keepdim=True) + 1e-8)
        assert torch.allclose(panels["val/inputs"], expected)
        # The blast radius half: an inputs-only failure must not take these out.
        assert "val/predictions" in panels
        assert "val/targets" in panels
        mock.metrics_service.save_images_batch.assert_called_once()

    @pytest.mark.unit
    def test_panel_and_recorder_read_one_rendering(self) -> None:
        """One owner (non-negotiable 17): both consumers get the same values.

        They used to convert ``inputs`` independently, which is why both were
        wrong the same two ways -- and why fixing one would have left the other.
        """
        import spectramr.infrastructure.training.strategies.mixins.metrics_mixin as mm

        mock = _seam_mock()
        inputs = torch.rand(1, 2, 16, 16)
        blank = torch.zeros(1, 1, 16, 16)

        with patch.object(mm, "feed_report_case_recorder") as feed:
            DiffusionTrainingStrategy._log_validation_images_to_tensorboard(
                mock,
                blank,
                blank.clone(),
                inputs,
                {},
                batch_idx=0,
                is_image_domain=True,
                inputs_are_image=True,
            )

        feed.assert_called_once()
        fed = feed.call_args.kwargs["inputs"]
        panel = _panels(mock)["val/inputs"]
        assert torch.equal(fed[: panel.shape[0]], panel)

    @pytest.mark.unit
    def test_input_is_not_rendered_when_no_consumer_wants_it(self) -> None:
        """The gate. ``inputs`` is TOUCHED only if a consumer will read it.

        Hoisting the render out of ``if log_input:`` made it unconditional,
        which put an inputs-only failure inside the method-wide ``except`` for
        every arm -- including the ones that never asked for an input panel.
        The sentinel raises on any tensor operation, so if the gate is removed
        this test loses the predictions and targets panels too.
        """

        class _Explodes:
            def __getattr__(self, name):
                raise AssertionError(f"inputs was touched (.{name}) with no consumer")

        mock = _seam_mock(log_input=False, recorder=False)
        blank = torch.zeros(1, 1, 16, 16)

        DiffusionTrainingStrategy._log_validation_images_to_tensorboard(
            mock,
            blank,
            blank.clone(),
            _Explodes(),
            {},
            batch_idx=0,
            is_image_domain=True,
            inputs_are_image=True,
        )

        panels = _panels(mock)
        assert "val/inputs" not in panels
        assert "val/predictions" in panels
        mock.metrics_service.save_images_batch.assert_called_once()


class TestValidationCallSitePassesTheScaledInputs:
    """Structural pins on the producer side of the seam.

    ``getsource`` substring checks are prose-satisfiable -- a comment naming
    ``log_inputs`` would score them green (and this file's #927 block relies on
    exactly that property in the opposite direction). These parse the AST, so
    only the real argument and the real assignment can satisfy them.
    """

    @staticmethod
    def _tree():
        from spectramr.infrastructure.training.strategies import diffusion as mod

        with open(inspect.getfile(mod), encoding="utf-8") as fh:
            return ast.parse(fh.read())

    @classmethod
    def _call_node(cls):
        tree = cls._tree()
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_log_validation_images_to_tensorboard"
        ]
        assert len(calls) == 1, f"expected one call site, found {len(calls)}"
        return calls[0]

    @pytest.mark.unit
    def test_third_positional_argument_is_the_scaled_inputs(self) -> None:
        node = self._call_node()
        third = node.args[2]
        assert isinstance(third, ast.Name), ast.dump(third)
        assert third.id == "log_inputs", (
            "the inputs argument must be the seam-crossed ``log_inputs``; "
            f"got {third.id!r} (``input_batch`` is the pre-seam raw tensor)"
        )

    @pytest.mark.unit
    def test_inputs_domain_flag_is_the_targets_element_not_the_preds_one(self) -> None:
        node = self._call_node()
        kw = {k.arg: k.value for k in node.keywords}
        assert "inputs_are_image" in kw, "the inputs domain flag is not passed at all"
        assert isinstance(kw["inputs_are_image"], ast.Name)
        assert kw["inputs_are_image"].id == "inputs_are_image", (
            "``inputs`` comes off the dataloader: its flag is the *targets* "
            "element of needs_ifft_for_visualization, never ``is_preds_image``"
        )
        # ...and the two must not be the same variable.
        assert kw["is_image_domain"].id == "is_preds_image"

    @pytest.mark.unit
    def test_log_inputs_is_denom_scaled_like_predictions_and_targets(self) -> None:
        """``log_inputs = input_batch * denom_scale`` -- structurally.

        The three tensors are rendered side by side, so they must share one
        scale. A ``log_inputs = input_batch`` regression is the original #1684.
        """
        tree = self._tree()
        assigns = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "log_inputs"
        ]
        assert len(assigns) == 1, f"expected one ``log_inputs`` assignment, found {len(assigns)}"
        value = assigns[0].value
        assert isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult), ast.dump(value)
        operands = {n.id for n in (value.left, value.right) if isinstance(n, ast.Name)}
        assert operands == {"input_batch", "denom_scale"}, operands

    @pytest.mark.unit
    def test_input_batch_is_decompressed_for_view(self) -> None:
        """The other half of the seam: undo ``log1p`` before scaling, like the target."""
        scored = _scored_against_the_target(self._tree())
        assert {"target_batch", "input_batch"} <= scored, (
            "``input_batch`` never passes through the target's decompression decision"
        )


class TestNonNegotiableThreeAndSeventeenInValidation:
    """Two structural pins on the same method, both in the #1684 change set.

    Neither is reachable behaviourally without standing up a real generator and
    a real metrics computer inside a ~400-line method, so both parse the AST.
    Prose cannot satisfy them, and a substring grep for ``_unscaled`` would be
    satisfied by the comment documenting its removal -- the exact trap this
    file's #927 block calls out in the opposite direction.
    """

    @staticmethod
    def _tree():
        from spectramr.infrastructure.training.strategies import diffusion as mod

        with open(inspect.getfile(mod), encoding="utf-8") as fh:
            return ast.parse(fh.read())

    def test_declared_multistep_sampling_raises_rather_than_degrading(self) -> None:
        """Non-negotiable 3.

        The arm DECLARED ``validation.sampling.enable_multistep_cold``. Falling
        through to the single-step forward measures a different quantity and is
        indistinguishable in the logs from the declared path working. Census
        2026-09-01: all 55 arms setting the flag resolve to
        ``KSpaceColdDiffusionGenerator``, which defines ``sample()`` -- so the
        raise fires on none of them.
        """
        guards = [
            n
            for n in ast.walk(self._tree())
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.BoolOp)
            and isinstance(n.test.op, ast.And)
            and any(isinstance(v, ast.Name) and v.id == "_multistep" for v in n.test.values)
            and any(
                isinstance(v, ast.UnaryOp)
                and isinstance(v.op, ast.Not)
                and isinstance(v.operand, ast.Call)
                and isinstance(v.operand.func, ast.Name)
                and v.operand.func.id == "hasattr"
                for v in n.test.values
            )
        ]
        assert len(guards) == 1, "the multistep capability guard moved or was removed"
        assert any(isinstance(s, ast.Raise) for s in guards[0].body), (
            "a missing ``sample()`` on a declared multistep arm must RAISE, "
            "never fall through to the single-step forward"
        )
        assert guards[0].orelse == [], "the guard must have no fallback branch"

    def test_no_validation_metric_key_is_minted_twice(self) -> None:
        """Non-negotiable 17.

        ``val_{k}_unscaled`` was a byte-for-byte alias of ``val_{k}`` -- same
        ``v``, same loop -- and the DDP all-reduce packs ``sorted(keys)``
        positionally, so every duplicate was a real collective payload.
        """
        suffixed = []
        for node in ast.walk(self._tree()):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if not (
                    isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "metrics"
                    and isinstance(tgt.slice, ast.JoinedStr)
                ):
                    continue
                literal = "".join(p.value for p in tgt.slice.values if isinstance(p, ast.Constant))
                if literal.endswith("_unscaled"):
                    suffixed.append((node.lineno, literal))
        assert suffixed == [], f"duplicate metric keys re-introduced: {suffixed}"


# ---------------------------------------------------------------------------
# Zero-filled baseline (Item C of the experiment_11_attention_none fix plan).
#
# ``experiment_11_attention_none`` scored BELOW a zero-filled reconstruction at
# every acceleration rung, on two independent clusters, with every dashboard
# green -- because nothing in the framework computed the zero-filled number.
# The baseline closes that: ``val_zf_<k>`` is the measurement the model was
# actually handed, graded against the SAME target, through the SAME transform,
# on the SAME data range, and ``val_zf_delta_<k>`` is the signed
# ``model - baseline``.
#
# Every leg below is a way the number can come out plausible and wrong, so each
# is pinned rather than reasoned about.
# ---------------------------------------------------------------------------


def _zf_mock(
    *,
    log_scaled: bool = False,
    measurement: torch.Tensor | None = None,
    metrics: tuple[str, ...] = ("psnr",),
) -> MagicMock:
    """A ``self`` complete enough to run ``_compute_validation_metrics`` to return.

    ``output_transform="none"`` makes ``_apply_metric_transforms`` the identity,
    which is the honest choice here: it isolates the seam under test (the
    decompress / denormalise / data-range legs) from the transform's own
    behaviour, and it is a real configuration the corpus uses.
    """
    from spectramr.config.schemas.data import DataProcessingConfigSchema
    from spectramr.core.metrics.computer import ValidationMetricsComputer
    from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig

    # Log compression runs inside the k-space normalization (#2353), so a
    # log-scaled arm has both on; the unit scale passed by the caller keeps the
    # denormalisation leg an exact identity.
    mock = MagicMock()
    mock.config = SimpleNamespace(
        data=SimpleNamespace(
            processing=DataProcessingConfigSchema(
                enable_kspace_normalization=log_scaled, enable_log_scaling=log_scaled
            )
        ),
        model=SimpleNamespace(input_type="image", model_type="kspace_cold_diffusion"),
        validation=SimpleNamespace(
            scoring=SimpleNamespace(
                enable_image_metrics=True, domain="image", output_transform="none"
            )
        ),
    )
    mock._is_cold_diffusion = MagicMock(return_value=True)
    mock._apply_metric_transforms = lambda pred, target, cfg: (pred, target)
    # Read late so a test that swaps ``_apply_metric_transforms`` still sees
    # the call the validation path makes through the views hook.
    mock._metric_transform_views = lambda pred, target, cfg: MetricViews(
        *mock._apply_metric_transforms(pred, target, cfg)
    )
    mock._measure_prediction_scale = lambda pred, target: {
        "pred_above_target_fraction": 0.0,
        "target_abs_max": 1.0,
        "pred_target_scale_ratio": 1.0,
    }
    mock._convert_metrics_to_floats = lambda d: d
    mock._PRED_SCALE_WARN_FRACTION = DiffusionTrainingStrategy._PRED_SCALE_WARN_FRACTION
    mock._ZF_BASELINE_METRICS = DiffusionTrainingStrategy._ZF_BASELINE_METRICS
    mock.validation_metrics_computer = ValidationMetricsComputer(
        ValidationMetricsConfig(
            metrics=[MetricSpec(name=n) for n in metrics], primary_metric=metrics[0]
        ),
        device="cpu",
    )
    mock._zf_measurement = measurement
    return mock


def _run_validation_metrics(mock, pred, target, inputs) -> dict:
    return DiffusionTrainingStrategy._compute_validation_metrics(
        mock,
        pred,
        target,
        inputs,
        torch.zeros(pred.shape[0], dtype=torch.long),
        None,
        torch.ones(pred.shape[0]),
    )


def _reference_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR through the same registry entry the production path resolves.

    Hand-rolling ``20*log10(...)`` here would make this test a second owner of
    the metric definition (non-negotiable 17): it would keep passing while the
    registered metric drifted away from it.
    """
    from spectramr.core.metrics.registry import MetricsRegistry

    metric = MetricsRegistry.get("psnr", device="cpu")
    data_range = target.abs().max().item()
    return float(metric(pred, target, data_range=data_range))


class TestMeasurementAwareMetricsReachTheProductionPath:
    """Observed firing end to end, not inferred from the wiring (non-negotiable 16).

    Three separate defects kept these columns out of every run (2026-09-16): the
    declared tuple named two of five, ``only=`` intersected against
    ``validation.scoring.compute`` rather than the ``metrics.compute`` the seam's
    comment named, and the context-free write-back overwrote the result with its
    own NaN. The unit tests in ``test_diffusion_measurement_metrics.py`` pin each
    mechanism; this one drives ``_compute_validation_metrics`` itself, because
    that is where all three met.
    """

    @staticmethod
    def _mock(names: list[str], mask: torch.Tensor | None):
        measurement = torch.rand(2, 2, 8, 8)
        mock = _zf_mock(measurement=measurement, metrics=names)
        # The real seam, bound to the mock host: a MagicMock would stub it out
        # and the test would observe nothing while claiming to observe firing.
        mock._measurement_aware_metrics = (
            lambda p, t_, m_, c, mk=None: DiffusionTrainingStrategy._measurement_aware_metrics(
                mock, p, t_, m_, c, mk
            )
        )
        mock._MEASUREMENT_AWARE_METRICS = DiffusionTrainingStrategy._MEASUREMENT_AWARE_METRICS
        mock._select_batch_compatible_smaps = lambda batch: None  # single-coil surrogate
        mock._rung_mask = mask
        return mock

    def test_the_columns_are_finite_when_the_arm_asks_for_them(self) -> None:
        torch.manual_seed(0)
        wanted = list(DiffusionTrainingStrategy._MEASUREMENT_AWARE_METRICS)
        mask = torch.zeros(2, 1, 8, 8)
        mask[..., 2:6, :] = 1.0
        mock = self._mock(["psnr", *wanted], mask)

        out = _run_validation_metrics(
            mock, torch.rand(2, 2, 8, 8), torch.rand(2, 2, 8, 8), torch.rand(2, 2, 8, 8)
        )

        for name in wanted:
            assert f"val_{name}" in out, f"val_{name} never reached the metrics dict"
            assert math.isfinite(out[f"val_{name}"]), (
                f"val_{name} is NaN -- the context-free pass won the write-back, "
                f"which is the defect merge_metric_passes exists to stop"
            )

    def test_the_baseline_crosses_the_same_seam_and_mints_a_delta(self) -> None:
        """``val_zf_delta_fabrication_excess`` is the number the cohort reports.

        The raw value carries a coil-subspace bias the delta cancels, so the seam
        has to score the zero-filled baseline too -- not only the prediction.
        """
        torch.manual_seed(0)
        mask = torch.zeros(2, 1, 8, 8)
        mask[..., 2:6, :] = 1.0
        mock = self._mock(["psnr", "fabrication_excess"], mask)

        out = _run_validation_metrics(
            mock, torch.rand(2, 2, 8, 8), torch.rand(2, 2, 8, 8), torch.rand(2, 2, 8, 8)
        )

        assert math.isfinite(out["val_fabrication_excess"])
        assert math.isfinite(out["val_zf_fabrication_excess"])
        assert out["val_zf_delta_fabrication_excess"] == pytest.approx(
            out["val_fabrication_excess"] - out["val_zf_fabrication_excess"], rel=1e-6
        )

    def test_an_arm_that_does_not_ask_gets_no_columns_at_all(self) -> None:
        """PLANTED: the shipped cohort config. Silence, not NaN -- which is why
        a cohort review that read the code did not catch it."""
        torch.manual_seed(0)
        mask = torch.zeros(2, 1, 8, 8)
        mask[..., 2:6, :] = 1.0
        mock = self._mock(["psnr", "mse"], mask)

        out = _run_validation_metrics(
            mock, torch.rand(2, 2, 8, 8), torch.rand(2, 2, 8, 8), torch.rand(2, 2, 8, 8)
        )

        assert "val_psnr" in out
        for name in DiffusionTrainingStrategy._MEASUREMENT_AWARE_METRICS:
            assert f"val_{name}" not in out

    def test_ndcr_scores_multicoil_kspace_per_coil(self) -> None:
        """#2316, observed through ``_compute_validation_metrics``.

        Four coils whose noise is i.i.d. per coil, so part of it lies off the
        sensitivity subspace. The zero-filled baseline and a prediction equal to
        the fully-sampled target both agree with the measurement on every
        acquired sample, so ``ndcr`` is 0 for both. Before the seam handed over
        ``pred_kspace`` the coil-combined images were re-projected through the
        maps and both read the off-subspace noise instead (0.29 for the model here).
        """
        g = torch.Generator().manual_seed(0)
        b, c, h, w = 2, 4, 8, 8
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        maps = torch.stack(
            [(0.6 + 0.4 * torch.cos(0.5 * (i + 1) * xx)) * torch.exp(1j * 0.5 * (i + 1) * (xx + yy))
             for i in range(c)]
        ).to(torch.complex64)
        maps = (maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()).expand(b, c, h, w)
        img = torch.exp(-(xx**2 + yy**2) / 0.3).to(torch.complex64).expand(b, 1, h, w)
        clean = fft2c(img * maps)
        noise = 0.05 * float(clean.abs().max()) * torch.complex(
            torch.randn(clean.shape, generator=g), torch.randn(clean.shape, generator=g)
        )
        full = clean + noise
        mask = torch.zeros(b, 1, h, w)
        mask[..., ::2] = 1.0
        mask[..., 3:5] = 1.0

        def layout(k: torch.Tensor) -> torch.Tensor:
            return torch.stack([k.real, k.imag], dim=2).flatten(1, 2)  # [r0, i0, r1, i1, ...]

        measured = layout(full * mask)
        mock = self._mock(["psnr", "ndcr"], mask)
        mock._zf_measurement = measured
        mock._select_batch_compatible_smaps = lambda batch: maps

        out = _run_validation_metrics(mock, layout(full), layout(full), measured)

        assert out["val_ndcr"] == pytest.approx(0.0, abs=1e-5)
        assert out["val_zf_ndcr"] == pytest.approx(0.0, abs=1e-5)


class TestPredictionAndTargetShareOneLevel:
    """The prediction is graded on the target's level, whatever its own magnitude.

    Per-tensor decompression left a prediction whose compressed |k|max passes
    ``DECOMPRESS_MAGNITUDE_CEILING`` in compressed units while its target was
    decompressed, so the metric compared two different scales.
    """

    @staticmethod
    def _pair() -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(11)
        target = torch.log1p(torch.rand(2, 2, 8, 8) * 50.0)
        pred = target.clone()
        pred[:, :, 0, 0] = DECOMPRESS_MAGNITUDE_CEILING + 5.0  # one diverged bin
        return pred, target

    def test_a_prediction_past_the_ceiling_is_decompressed_with_its_target(self) -> None:
        pred, target = self._pair()
        mock = _zf_mock(log_scaled=True)
        seen = {}
        mock._apply_metric_transforms = lambda p, t, cfg: (seen.update(p=p, t=t) or (p, t))
        _run_validation_metrics(mock, pred, target, target.clone())
        expected_pred = decompress_kspace_log(pred, channel_dim=1)
        expected_target = decompress_kspace_log(target, channel_dim=1)
        torch.testing.assert_close(seen["t"], expected_target)
        torch.testing.assert_close(seen["p"], expected_pred)

    def test_per_tensor_decisions_would_have_split_the_pair(self) -> None:
        """Planted: the old rule leaves this prediction compressed, its target not."""
        pred, target = self._pair()
        assert decompress_for_view(pred, log_scaled=True, channel_dim=1) is pred
        assert decompress_for_view(target, log_scaled=True, channel_dim=1) is not target

    def test_log_scaling_without_normalization_is_scored_uncompressed(self) -> None:
        """#2353: ``enable_log_scaling`` alone compresses nothing, so validation
        must not ``expm1`` the pair on the bare declaration."""
        from spectramr.config.schemas.data import DataProcessingConfigSchema

        pred, target = self._pair()
        mock = _zf_mock()
        mock.config.data.processing = DataProcessingConfigSchema(
            enable_kspace_normalization=False, enable_log_scaling=True
        )
        seen = {}
        mock._apply_metric_transforms = lambda p, t, cfg: (seen.update(p=p, t=t) or (p, t))
        _run_validation_metrics(mock, pred, target, target.clone())
        torch.testing.assert_close(seen["t"], target)
        torch.testing.assert_close(seen["p"], pred)

    def test_a_perfect_prediction_scores_as_perfect(self) -> None:
        _, target = self._pair()
        out = _run_validation_metrics(
            _zf_mock(log_scaled=True), target.clone(), target, target.clone()
        )
        assert out["val_psnr"] == pytest.approx(
            _reference_psnr(
                decompress_kspace_log(target, channel_dim=1),
                decompress_kspace_log(target, channel_dim=1),
            ),
            rel=1e-6,
        )


class TestZeroFilledBaselineIsEmittedAndComparable:
    def test_the_baseline_keys_are_emitted_through_the_production_path(self) -> None:
        """Observed firing, not inferred from the wiring (non-negotiable 16)."""
        torch.manual_seed(0)
        mock = _zf_mock(measurement=torch.rand(2, 1, 8, 8))
        out = _run_validation_metrics(
            mock, torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
        )
        assert "val_zf_psnr" in out
        assert "val_zf_delta_psnr" in out
        assert math.isfinite(out["val_zf_psnr"])

    def test_the_baseline_grades_the_measurement_not_the_target(self) -> None:
        """The idealised-baseline defect, stated as a number.

        A target-derived ZF is noise-free and unreachable: it would score
        ABOVE the model on a healthy run and hide the very "worse than
        zero-filled" verdict this baseline exists to surface. Here the
        measurement, the input and the target are three distinct tensors, so
        only the right source reproduces the reported value.
        """
        torch.manual_seed(1)
        measurement = torch.rand(2, 1, 8, 8)
        target = torch.rand(2, 1, 8, 8)
        mock = _zf_mock(measurement=measurement)
        out = _run_validation_metrics(mock, torch.rand(2, 1, 8, 8), target, torch.rand(2, 1, 8, 8))

        assert out["val_zf_psnr"] == pytest.approx(_reference_psnr(measurement, target), abs=1e-4)
        assert out["val_zf_psnr"] != pytest.approx(_reference_psnr(target, target), abs=1e-4)

    def test_the_delta_is_the_signed_model_minus_baseline(self) -> None:
        """Signed, not oriented: ``metric_higher_is_better`` resolves
        ``val_zf_delta_hfen`` through ``hfen`` to "lower is better", which is
        correct for a raw difference and backwards for an oriented gain.
        """
        torch.manual_seed(2)
        mock = _zf_mock(measurement=torch.rand(2, 1, 8, 8))
        out = _run_validation_metrics(
            mock, torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
        )
        assert out["val_zf_delta_psnr"] == pytest.approx(
            out["val_psnr"] - out["val_zf_psnr"], abs=1e-6
        )

    def test_the_delta_direction_resolves_correctly_for_both_metrics(self) -> None:
        """The naming choice, executed against the real direction resolver."""
        from spectramr.core.metrics.metric_directions import metric_higher_is_better

        assert metric_higher_is_better("val_zf_delta_psnr") is True
        assert metric_higher_is_better("val_zf_delta_hfen") is False
        assert metric_higher_is_better("val_zf_psnr") is True
        assert metric_higher_is_better("val_zf_hfen") is False

    def test_the_baseline_crosses_the_log_decompression_seam(self) -> None:
        """``log1p`` is non-linear, so grading compressed k-space grades a
        different signal. This is #1684's lesson applied to a number.

        TWO channels, not one, and the non-vacuity assertion below is why.
        ``decompress_for_view`` gates on ``shape[channel_dim] % 2 == 0`` (it
        only ever undoes compression of real-stacked complex k-space), so on a
        1-channel tensor it returns its input untouched -- and BOTH sides of
        the comparison would then be the compressed tensor, agreeing perfectly
        while the seam under test never ran. That is the shape this very test
        had until a planted "skip the decompression" violation failed to turn
        it red.
        """
        torch.manual_seed(3)
        compressed = torch.log1p(torch.rand(2, 2, 8, 8) * 50.0)
        # The target is compressed by the same transform as the measurement, as
        # on the data path; it is graded decompressed.
        target_linear = torch.rand(2, 2, 8, 8) * 50.0
        target = compress_kspace_log(target_linear, channel_dim=1)
        decompressed = decompress_for_view(compressed, log_scaled=True, channel_dim=1)
        assert not torch.allclose(decompressed, compressed), (
            "decompression was a no-op on this fixture, so the assertion below "
            "would hold whether or not the production path decompresses"
        )

        mock = _zf_mock(log_scaled=True, measurement=compressed)
        out = _run_validation_metrics(
            mock,
            torch.log1p(torch.rand(2, 2, 8, 8) * 50.0),
            target,
            torch.log1p(torch.rand(2, 2, 8, 8) * 50.0),
        )
        graded_target = decompress_for_view(target, log_scaled=True, channel_dim=1)
        assert out["val_zf_psnr"] == pytest.approx(
            _reference_psnr(decompressed, graded_target), abs=1e-3
        )

    def test_no_baseline_keys_when_no_measurement_was_recorded(self) -> None:
        """A non-cold arm never records one, so it emits none -- on EVERY batch
        and rank, which is what the DDP all-reduce's sorted-key packing needs
        (#1690). An absent baseline must never become a defaulted one.
        """
        torch.manual_seed(4)
        mock = _zf_mock(measurement=None)
        out = _run_validation_metrics(
            mock, torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
        )
        assert [k for k in out if k.startswith("val_zf")] == []

    def test_the_measurement_is_popped_so_a_later_rung_cannot_inherit_it(self) -> None:
        """Read-and-clear. The cascade runs rung after rung on one strategy
        instance, so a stash that survived its consumer would report rung N-1's
        measurement under rung N's acceleration -- plausible, and wrong.
        """
        torch.manual_seed(5)
        mock = _zf_mock(measurement=torch.rand(2, 1, 8, 8))
        _run_validation_metrics(
            mock, torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
        )
        assert mock._zf_measurement is None

    def test_the_baseline_is_restricted_to_the_cheap_pair(self) -> None:
        """An arm grading on a perceptual metric must not pay for it twice."""
        torch.manual_seed(6)
        mock = _zf_mock(measurement=torch.rand(2, 1, 8, 8), metrics=("psnr", "mse"))
        out = _run_validation_metrics(
            mock, torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
        )
        assert "val_mse" in out, "the arm's own metric set must be unaffected"
        assert "val_zf_mse" not in out
        assert "val_zf_psnr" in out


def _method_tree(method) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]


def _compute_calls(tree: ast.AST) -> list[ast.Call]:
    """Every ``<something>.compute(...)`` call in the tree, in source order."""
    return sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compute"
        ),
        key=lambda n: n.lineno,
    )


def _kw(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


class TestZeroFilledBaselineWiring:
    """Structural pins for the legs that fail SILENTLY.

    A behavioural test can catch a baseline computed from the wrong tensor,
    because the number changes. It cannot catch a ``denom_scale`` that was
    dropped -- PSNR normalises by the target's own peak, so a common scale
    factor cancels exactly, and ``val_zf_psnr`` stays right up to the moment
    an arm grades on a scale-sensitive metric. Those legs are pinned here.
    """

    def test_the_stash_is_the_masked_input_not_the_target(self) -> None:
        tree = _method_tree(DiffusionTrainingStrategy._generate_validation_prediction)
        sources = [
            node.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Name)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "_zf_measurement" for t in node.targets
            )
        ]
        assert sources == ["masked_input"], (
            f"the zero-filled baseline is sourced from {sources}; anything but "
            "`masked_input` grades an idealised measurement the model never saw"
        )

    def test_the_producer_clears_the_stash_unconditionally(self) -> None:
        """At the FUNCTION body's top level, not inside the ``try`` or the cold
        branch -- a clear that a failing generation can skip is how rung N-1's
        measurement reaches rung N.
        """
        tree = _method_tree(DiffusionTrainingStrategy._generate_validation_prediction)
        top_level_clears = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
            and any(
                isinstance(t, ast.Attribute) and t.attr == "_zf_measurement" for t in node.targets
            )
        ]
        assert len(top_level_clears) == 1

    def test_the_consumer_pops_rather_than_reads(self) -> None:
        tree = _method_tree(DiffusionTrainingStrategy._compute_validation_metrics)
        clears = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
            and any(
                isinstance(t, ast.Attribute) and t.attr == "_zf_measurement" for t in node.targets
            )
        ]
        assert len(clears) == 1, "the consumer must clear what it read"

    def test_the_baseline_is_decompressed_before_it_is_graded(self) -> None:
        tree = _method_tree(DiffusionTrainingStrategy._compute_validation_metrics)
        decompressed = _scored_against_the_target(tree)
        assert "_zf_measurement" in decompressed, (
            "the baseline skipped the log-decompression seam; an inverse FFT of "
            "compressed k-space reconstructs a different signal (#1684)"
        )

    def test_the_baseline_is_denormalised_by_the_shared_scale(self) -> None:
        tree = _method_tree(DiffusionTrainingStrategy._compute_validation_metrics)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "zf_for_metrics" for t in node.targets)
                and isinstance(node.value, ast.BinOp)
            ):
                assert isinstance(node.value.op, ast.Mult)
                operands = {
                    n.id for n in (node.value.left, node.value.right) if isinstance(n, ast.Name)
                }
                assert operands == {"_zf_measurement", "denom_scale"}
                return
        raise AssertionError("`zf_for_metrics` is not built by scaling the measurement")

    def test_the_subset_compute_runs_before_the_full_one(self) -> None:
        """``compute`` clears ``last_not_applicable`` per call and
        ``training_loop.py`` reads it afterwards, so a subset call placed second
        would narrow the N/A report to psnr/hfen.
        """
        calls = _compute_calls(_method_tree(DiffusionTrainingStrategy._compute_validation_metrics))
        subset = [c for c in calls if _kw(c, "only") is not None]
        full = [c for c in calls if _kw(c, "only") is None]
        assert len(subset) == 1 and len(full) == 1
        assert subset[0].lineno < full[0].lineno

    def test_the_subset_is_the_declared_constant(self) -> None:
        calls = _compute_calls(_method_tree(DiffusionTrainingStrategy._compute_validation_metrics))
        only = _kw(next(c for c in calls if _kw(c, "only") is not None), "only")
        assert isinstance(only, ast.Attribute) and only.attr == "_ZF_BASELINE_METRICS"

    def test_both_computes_share_one_target_and_one_data_range(self) -> None:
        """A baseline re-measured against its own peak, or transformed into its
        own target, is a PSNR on a different axis wearing the same name.
        """
        calls = _compute_calls(_method_tree(DiffusionTrainingStrategy._compute_validation_metrics))
        zf = next(c for c in calls if _kw(c, "only") is not None)
        primary = next(c for c in calls if _kw(c, "only") is None)

        # The primary pass scores the transform's views, whose magnitude target
        # is the one bound to ``target_transformed`` (#867); the ZF subset names it.
        assert isinstance(zf.args[1], ast.Name) and zf.args[1].id == "target_transformed"
        assert isinstance(primary.func.value, ast.Name) and primary.func.value.id == "metric_views"
        for call in (zf, primary):
            data_range = _kw(call, "data_range")
            assert isinstance(data_range, ast.Name) and data_range.id == "dynamic_data_range"


def _clears_the_stash(node: ast.AST) -> bool:
    """True if ``self._zf_measurement = None`` appears anywhere in ``node``."""
    for assign in ast.walk(node):
        if not isinstance(assign, ast.Assign):
            continue
        if not (isinstance(assign.value, ast.Constant) and assign.value.value is None):
            continue
        for tgt in assign.targets:
            if (
                isinstance(tgt, ast.Attribute)
                and tgt.attr == "_zf_measurement"
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"
            ):
                return True
    return False


def _calls_self(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == name
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == "self"
        for c in ast.walk(node)
    )


class TestEveryProducerCallSiteHasAConsumerOrADiscard:
    """The stash is a *retention* hazard, not a correctness one.

    ``_generate_validation_prediction`` clears unconditionally on entry, so a
    stale measurement can never be read as another rung's baseline. What it can
    do is stay referenced on the strategy instance for hours of training --
    ``get_validation_images`` and ``_maybe_hallucination_metrics`` both drive
    the producer and neither calls the consumer, so the last masked k-space
    batch of a validation pass survives until the *next* validation's first
    generate. That is a full batch of complex k-space on a 16 GB V100 with a
    documented val-time-OOM history.

    Pinning the two known sites by name would go blind the moment a third
    caller appears, so this walks the class instead: every method that reaches
    the producer must either reach the consumer (which pops) or discard.
    """

    @staticmethod
    def _producer_callers() -> dict[str, ast.FunctionDef]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(DiffusionTrainingStrategy))).body[0]
        assert isinstance(tree, ast.ClassDef)
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name != "_generate_validation_prediction"
            and _calls_self(node, "_generate_validation_prediction")
        }

    def test_the_walk_actually_reaches_the_known_callers(self) -> None:
        """Non-vacuity: an empty set would make every assertion below pass."""
        callers = self._producer_callers()
        assert {"get_validation_images", "_maybe_hallucination_metrics"} <= set(callers)

    def test_every_producer_caller_pops_or_discards(self) -> None:
        offenders = [
            name
            for name, node in self._producer_callers().items()
            if not (_calls_self(node, "_compute_validation_metrics") or _clears_the_stash(node))
        ]
        assert offenders == [], (
            f"{offenders} drive _generate_validation_prediction without popping the "
            "zero-filled stash; add `self._zf_measurement = None` before returning."
        )

    def test_the_image_path_discards_before_its_early_return(self) -> None:
        """``get_validation_images`` returns ``None`` when generation failed; a
        discard placed only before the final ``return`` misses that path.
        """
        node = self._producer_callers()["get_validation_images"]
        last_generate = max(
            c.lineno
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "_generate_validation_prediction"
        )
        returns = [
            r.lineno
            for r in ast.walk(node)
            if isinstance(r, ast.Return) and r.lineno > last_generate
        ]
        clears = [
            a.lineno
            for a in ast.walk(node)
            if isinstance(a, ast.Assign) and _clears_the_stash(a) and a.lineno > last_generate
        ]
        assert returns, "no return follows the generate block -- the pin is vacuous"
        assert clears, "no discard follows the generate block in get_validation_images"
        assert min(clears) < min(returns), (
            "the discard sits below an earlier return; that path leaks the stash"
        )

    def test_the_diagnostic_discards_on_every_exit_path(self) -> None:
        """``_maybe_hallucination_metrics`` swallows exceptions and returns from
        three places; only a ``finally`` covers all of them.
        """
        node = self._producer_callers()["_maybe_hallucination_metrics"]
        tries = [t for t in ast.walk(node) if isinstance(t, ast.Try)]
        assert any(any(_clears_the_stash(stmt) for stmt in t.finalbody) for t in tries), (
            "the discard is not in a finally: block"
        )


class TestDiffusionDeclaredMetricKeys:
    """#1682 -- ``pre_dc_kspace_l1`` gets a column exactly when it is stamped.

    ``_add_pre_dc_fidelity`` stamps the key on BOTH of its ``lam > 0`` paths
    (the active term, and the INACTIVE sentinel that fires when the generator
    exposed no pre-DC tuple) and on neither when ``lam <= 0``. The declaration
    must track that gate exactly: declaring more promises an always-empty
    column, declaring less resumes discarding the only observable of the
    terminal rung's only gradient.
    """

    @staticmethod
    def _stub(lam):
        from types import SimpleNamespace

        return SimpleNamespace(
            config=SimpleNamespace(
                losses=SimpleNamespace(reconstruction=SimpleNamespace(lambda_pre_dc_kspace=lam)),
                model=SimpleNamespace(model_kwargs={}),
            )
        )

    def test_declared_when_weight_is_positive(self):
        from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

        assert DiffusionTrainingStrategy.declared_metric_keys(self._stub(0.3)) == {
            "pre_dc_kspace_l1"
        }

    def test_not_declared_when_weight_is_zero_or_absent(self):
        from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

        assert DiffusionTrainingStrategy.declared_metric_keys(self._stub(0.0)) == frozenset()
        assert DiffusionTrainingStrategy.declared_metric_keys(self._stub(None)) == frozenset()

    def test_absent_losses_block_raises_rather_than_defaulting(self):
        """A missing SSOT block must NOT quietly resolve to "no column".

        `config.losses` is required; reading it through a `getattr` fallback
        would turn a malformed config into a silently-absent column, which is
        the same class of defect as the discard this change fixes. The producer
        raises there, so the declaration raises there too.
        """
        from types import SimpleNamespace

        import pytest

        from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

        bare = SimpleNamespace(config=SimpleNamespace(losses=None))
        with pytest.raises(AttributeError):
            DiffusionTrainingStrategy.declared_metric_keys(bare)

    def test_the_clamp_columns_are_declared_exactly_when_the_clamp_is_on(self):
        from types import SimpleNamespace

        from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

        def stub(model_kwargs):
            return SimpleNamespace(
                config=SimpleNamespace(
                    losses=SimpleNamespace(reconstruction=None),
                    model=SimpleNamespace(model_kwargs=model_kwargs),
                )
            )

        on = DiffusionTrainingStrategy.declared_metric_keys(stub({"output_kspace_clip_ratio": 1.3}))
        assert on == {"clip_bind_frac", "clip_overshoot_mean"}
        assert DiffusionTrainingStrategy.declared_metric_keys(stub({})) == frozenset()

    def test_dc_telemetry_columns_are_declared_from_the_generator(self):
        """The Wiener-gain DC arm discarded its trust/gain telemetry: no column."""
        from types import SimpleNamespace

        import torch

        from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

        emitted = {
            "noise_adaptive_trust/mean": torch.tensor(0.5),
            "wiener_gain/alpha": torch.tensor(1.0),
        }
        gen = SimpleNamespace(get_kan_trust_map_telemetry=lambda: emitted)
        stub = SimpleNamespace(
            config=SimpleNamespace(
                losses=SimpleNamespace(reconstruction=None),
                model=SimpleNamespace(model_kwargs={}),
            ),
            generator_model=gen,
        )
        assert DiffusionTrainingStrategy.declared_metric_keys(stub) == set(emitted)

    def test_declaration_gate_matches_the_producer_gate(self):
        """Pin the two ends of one condition together.

        The producer's gate is ``if lam <= 0.0: return total_loss`` inside
        ``_add_pre_dc_fidelity``. If that threshold moves without this
        declaration moving, the column and the value disagree again -- silently,
        and in whichever direction the drift happened to go.
        """
        from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

        assert DiffusionTrainingStrategy.declared_metric_keys(self._stub(1e-12)) == {
            "pre_dc_kspace_l1"
        }
        assert DiffusionTrainingStrategy.declared_metric_keys(self._stub(0.0)) == frozenset()


# ---------------------------------------------------------------------------
# Multi-sample cold-diffusion validation ensemble (review follow-up item 2, 2026-09).
#
# ``validation.sampling.ensemble_samples > 1`` draws N reverse samples of the
# same input through ``_sample_multistep_chunked``, one C6 seed offset each,
# grades their k-space mean, and scores the spread in the metric domain as
# ``val_ensemble_std_mean`` and ``val_empirical_coverage``. Every leg is a way
# the numbers can come out plausible and wrong, so each is pinned: the members
# must actually differ, the keys must come out of the production consumer, the
# computer's own N/A must not overwrite them, and the names must resolve.
# ---------------------------------------------------------------------------


class _StochasticGen:
    """``sample()`` adds ``sampler_sigma``-scaled noise from the member's own stream."""

    def __init__(self, sigma: float = 0.3, seed: int = 7) -> None:
        self.sampler_sigma = sigma
        self.sampler_seed = seed
        self.calls: list[dict] = []

    def sample(self, measurement, mask, inference_timesteps, smaps, start_timestep, seed_offset=0):
        self.calls.append({"seed_offset": seed_offset, "n": measurement.shape[0]})
        g = torch.Generator().manual_seed(self.sampler_seed + seed_offset)
        return measurement + self.sampler_sigma * torch.randn(measurement.shape, generator=g)


class _LegacyGen:
    """The pre-ensemble ``sample()`` signature: no ``seed_offset``, no ``**kwargs``."""

    def sample(self, measurement, mask, inference_timesteps, smaps, start_timestep):
        return measurement


def _ensemble_host(
    n: int = 4, chunk: int = 8, image_metrics: bool = True
) -> DiffusionTrainingStrategy:
    host = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
    host.config = SimpleNamespace(
        validation=SimpleNamespace(
            loader=SimpleNamespace(chunk_size=chunk),
            sampling=SimpleNamespace(
                ensemble_samples=n, coverage_k=2.0, enable_multistep_cold=True
            ),
            scoring=SimpleNamespace(enable_image_metrics=image_metrics),
        )
    )
    host.logging_service = MagicMock()
    return host


def _bind_ensemble_helpers(mock: MagicMock) -> None:
    """A MagicMock host auto-mocks the helpers; bind the real ones."""
    for name in ("_ensemble_std_in_metric_domain", "_ensemble_metrics"):
        setattr(mock, name, MethodType(getattr(DiffusionTrainingStrategy, name), mock))


class TestValidationEnsemble:
    def test_members_differ_and_member_zero_is_the_single_sample_call(self) -> None:
        gen = _StochasticGen()
        host = _ensemble_host()
        measurement = torch.randn(2, 4, 8, 8)

        members = host._sample_ensemble_chunked(gen, measurement, None, 3, 4)

        assert members.shape == (4, 2, 4, 8, 8)
        assert [c["seed_offset"] for c in gen.calls] == [0, 1, 2, 3]
        single = host._sample_multistep_chunked(gen, measurement, None, 3)
        assert torch.equal(members[0], single)
        assert not torch.equal(members[0], members[1])
        assert float(members.std(dim=0).mean()) > 0.0

    def test_a_zero_offset_is_never_forwarded(self) -> None:
        """Generators without the kwarg keep working; only a real member asks for it."""
        host = _ensemble_host()
        measurement = torch.randn(2, 4, 8, 8)
        host._sample_multistep_chunked(_LegacyGen(), measurement, None, 3)
        host._sample_multistep_chunked(_LegacyGen(), measurement, None, 3, seed_offset=0)
        with pytest.raises(TypeError, match="seed_offset"):
            host._sample_multistep_chunked(_LegacyGen(), measurement, None, 3, seed_offset=1)

    def test_the_ensemble_is_chunked_like_the_sampler(self) -> None:
        """Outer loop over members, inner ``val_chunk_size`` loop -- peak stays one chunk."""
        gen = _StochasticGen()
        host = _ensemble_host(chunk=1)

        members = host._sample_ensemble_chunked(gen, torch.randn(3, 2, 8, 8), None, 3, 2)

        assert [c["n"] for c in gen.calls] == [1, 1, 1, 1, 1, 1]
        assert [c["seed_offset"] for c in gen.calls] == [0, 0, 0, 1, 1, 1]
        assert members.shape == (2, 3, 2, 8, 8)

    def test_resolver_returns_one_and_says_the_ensemble_is_off(self) -> None:
        host = _ensemble_host(n=1)
        assert host._resolve_validation_ensemble(_StochasticGen()) == 1
        assert "ensemble is off" in host.logging_service.log_info.call_args.args[0]
        assert host.ensemble_provenance == {"ensemble_samples": 1}

    def test_resolver_stamps_provenance_and_logs_once(self) -> None:
        host = _ensemble_host(n=4)
        gen = _StochasticGen(sigma=0.3, seed=7)
        assert host._resolve_validation_ensemble(gen) == 4
        assert host._resolve_validation_ensemble(gen) == 4
        assert host.logging_service.log_info.call_count == 1
        assert host.ensemble_provenance == {
            "ensemble_samples": 4,
            "coverage_k": 2.0,
            "sampler_sigma": 0.3,
            "sampler_seed": 7,
            "seed_offsets": [0, 1, 2, 3],
        }

    def test_resolver_refuses_a_deterministic_sampler(self) -> None:
        """The planted facade: sigma 0 means N identical members."""
        with pytest.raises(ConfigurationError, match="sampler_sigma"):
            _ensemble_host(n=4)._resolve_validation_ensemble(_StochasticGen(sigma=0.0))

    def test_resolver_refuses_a_generator_that_cannot_prove_it_is_stochastic(self) -> None:
        with pytest.raises(ConfigurationError, match="sampler_sigma"):
            _ensemble_host(n=2)._resolve_validation_ensemble(SimpleNamespace())

    def test_resolver_refuses_an_ensemble_nothing_reads(self) -> None:
        with pytest.raises(ConfigurationError, match="enable_image_metrics"):
            _ensemble_host(n=2, image_metrics=False)._resolve_validation_ensemble(_StochasticGen())

    def test_the_keys_are_emitted_through_the_production_path(self) -> None:
        """The fires-test (non-negotiable 16): observed out of ``_compute_validation_metrics``."""
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.quantitative.conformal_risk import EmpiricalCoverageMetric

        torch.manual_seed(0)
        mock = _zf_mock()
        mock.config.validation.sampling = SimpleNamespace(coverage_k=2.0, ensemble_samples=4)
        _bind_ensemble_helpers(mock)
        target = torch.rand(2, 1, 8, 8)
        members = target.unsqueeze(0) + 0.1 * torch.randn(4, 2, 1, 8, 8)
        mock._ensemble_members = members

        out = _run_validation_metrics(mock, members.mean(0), target, torch.rand(2, 1, 8, 8))

        assert out["val_ensemble_std_mean"] > 0.0
        assert 0.0 < out["val_empirical_coverage"] <= 1.0
        assert mock._ensemble_members is None, "the stack must be popped, never inherited"
        # The number IS the registered metric on the same tensors (one owner).
        expected = EmpiricalCoverageMetric(k=2.0)(
            members.mean(0), target, context=MetricContext(ensemble_std=members.std(0))
        )
        assert out["val_empirical_coverage"] == pytest.approx(expected)
        assert out["val_ensemble_std_mean"] == pytest.approx(float(members.std(0).mean()))

    def test_no_keys_when_the_ensemble_is_off(self) -> None:
        mock = _zf_mock()
        mock.config.validation.sampling = SimpleNamespace(coverage_k=2.0, ensemble_samples=1)
        _bind_ensemble_helpers(mock)
        mock._ensemble_members = None

        out = _run_validation_metrics(
            mock, torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
        )

        assert "val_ensemble_std_mean" not in out
        assert "val_empirical_coverage" not in out

    def test_the_ensemble_value_wins_over_the_computers_not_applicable(self) -> None:
        """An arm that ALSO lists ``empirical_coverage`` in ``metrics.compute`` gets the
        computer's declared N/A (NaN) for it; the write-back loop must not overwrite
        the ensemble's value with that NaN."""
        torch.manual_seed(1)
        mock = _zf_mock(metrics=("psnr", "empirical_coverage"))
        mock.config.validation.sampling = SimpleNamespace(coverage_k=2.0, ensemble_samples=4)
        _bind_ensemble_helpers(mock)
        target = torch.rand(2, 1, 8, 8)
        members = target.unsqueeze(0) + 0.1 * torch.randn(4, 2, 1, 8, 8)
        mock._ensemble_members = members

        out = _run_validation_metrics(mock, members.mean(0), target, torch.rand(2, 1, 8, 8))

        assert not math.isnan(out["val_empirical_coverage"])
        assert 0.0 < out["val_empirical_coverage"] <= 1.0

    def test_the_std_crosses_the_log_decompression_seam(self) -> None:
        """Spread is measured where the residual is: decompressed, like the prediction."""
        torch.manual_seed(2)
        mock = _zf_mock(log_scaled=True)
        _bind_ensemble_helpers(mock)
        members = torch.log1p(torch.rand(3, 2, 2, 8, 8) * 50.0)
        target = torch.rand(2, 2, 8, 8) * 50.0

        std = mock._ensemble_std_in_metric_domain(
            members, torch.ones(2, 1, 1, 1), True, target, mock.config.validation
        )

        decompressed = torch.stack(
            [decompress_for_view(m, log_scaled=True, channel_dim=1) for m in members.unbind(0)]
        )
        assert not torch.allclose(decompressed, members), "fixture must exercise the seam"
        torch.testing.assert_close(std, decompressed.std(dim=0))

    def test_the_cascade_mean_covers_the_ensemble_keys(self) -> None:
        all_metrics = {
            "val_empirical_coverage_2x": 1.0,
            "val_empirical_coverage_8x": 0.5,
            "val_ensemble_std_mean_2x": 0.1,
            "val_ensemble_std_mean_8x": 0.3,
        }
        DiffusionTrainingStrategy._stamp_accel_mean(
            all_metrics, [2, 8], list(DiffusionTrainingStrategy._ENSEMBLE_METRICS)
        )
        assert all_metrics["val_empirical_coverage_mean"] == pytest.approx(0.75)
        assert all_metrics["val_ensemble_std_mean_mean"] == pytest.approx(0.2)
        # ...and the production call site hands them over.
        src = inspect.getsource(DiffusionTrainingStrategy.validation_step)
        assert "*self._ENSEMBLE_METRICS" in src

    def test_the_zero_filled_normalised_mean_puts_the_rungs_on_one_axis(self) -> None:
        all_metrics = {
            "val_hfen_2x": 0.2,
            "val_zf_hfen_2x": 0.4,
            "val_hfen_8x": 0.6,
            "val_zf_hfen_8x": 1.2,
            "val_hfen_32x": 1.5,
            "val_zf_hfen_32x": 1.0,
        }
        DiffusionTrainingStrategy._stamp_zf_normalised_mean(all_metrics, [2, 8, 32])
        assert all_metrics["val_hfen_vs_zf_mean"] == pytest.approx((0.5 + 0.5 + 1.5) / 3)
        assert "val_hfen_vs_zf_mean_partial" not in all_metrics

    @pytest.mark.parametrize(
        "drop",
        [
            {"val_zf_hfen_32x": None},  # baseline never computed on a rung
            {"val_zf_hfen_32x": 0.0},  # baseline not positive: the ratio is undefined
        ],
    )
    def test_a_rung_that_cannot_be_normalised_makes_the_mean_partial(self, drop) -> None:
        """A silently shortened mean would outscore the full one (#1303), so it changes name."""
        all_metrics = {
            "val_hfen_2x": 0.2,
            "val_zf_hfen_2x": 0.4,
            "val_hfen_32x": 1.5,
            "val_zf_hfen_32x": 1.0,
        }
        for key, value in drop.items():
            if value is None:
                all_metrics.pop(key)
            else:
                all_metrics[key] = value
        DiffusionTrainingStrategy._stamp_zf_normalised_mean(all_metrics, [2, 32])
        assert "val_hfen_vs_zf_mean" not in all_metrics
        assert all_metrics["val_hfen_vs_zf_mean_partial"] == pytest.approx(0.5)

    def test_the_band_metrics_are_normalised_by_their_zero_filled_baseline(self) -> None:
        all_metrics = {
            "val_unacquired_bin_nrmse_2x": 0.5,
            "val_zf_unacquired_bin_nrmse_2x": 1.0,
            "val_unacquired_bin_nrmse_8x": 0.9,
            "val_zf_unacquired_bin_nrmse_8x": 1.0,
        }
        DiffusionTrainingStrategy._stamp_zf_normalised_mean(
            all_metrics, [2, 8], metric="unacquired_bin_nrmse"
        )
        assert all_metrics["val_unacquired_bin_nrmse_vs_zf_mean"] == pytest.approx(0.7)
        emitted = DiffusionTrainingStrategy.capabilities.emitted_metrics
        assert "val_null_band_nrmse_outer_vs_zf_mean" in emitted

    def test_the_zero_filled_baseline_is_graded_on_the_robust_psnr_too(self) -> None:
        assert "robust_mri_psnr" in DiffusionTrainingStrategy._ZF_BASELINE_METRICS

    def test_an_incomplete_cascade_stamps_the_partial_name(self) -> None:
        all_metrics = {"val_hfen_2x": 0.2, "val_zf_hfen_2x": 0.4}
        DiffusionTrainingStrategy._stamp_zf_normalised_mean(all_metrics, [2], complete=False)
        assert "val_hfen_vs_zf_mean" not in all_metrics
        assert "val_hfen_vs_zf_mean_partial" in all_metrics

    def test_the_normalised_key_is_declared_ranked_lower_is_better_and_stamped_in_production(
        self,
    ) -> None:
        import spectramr.core.metrics  # noqa: F401  -- populate the registry
        from spectramr.core.metrics.metric_directions import metric_higher_is_better
        from spectramr.core.metrics.registry import MetricsRegistry
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            universal_validation_keys,
        )
        from spectramr.infrastructure.validation.witness.checks.validation_metric_checks import (
            unresolved_metric_names,
        )

        emitted = DiffusionTrainingStrategy.capabilities.emitted_metrics
        universal = universal_validation_keys(DiffusionTrainingStrategy)
        assert (
            unresolved_metric_names(
                ["val_hfen_vs_zf_mean"], MetricsRegistry.is_registered, emitted, universal=universal
            )
            == []
        )
        assert metric_higher_is_better("val_hfen_vs_zf_mean") is False
        src = inspect.getsource(DiffusionTrainingStrategy.validation_step)
        assert "self._stamp_zf_normalised_mean(" in src

    def test_the_acquired_bin_keys_resolve_per_rung_and_as_the_cascade_mean(self) -> None:
        import spectramr.core.metrics  # noqa: F401  -- populate the registry
        from spectramr.core.metrics.metric_directions import metric_higher_is_better
        from spectramr.core.metrics.registry import MetricsRegistry
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            universal_validation_keys,
        )
        from spectramr.infrastructure.validation.witness.checks.validation_metric_checks import (
            unresolved_metric_names,
        )

        emitted = DiffusionTrainingStrategy.capabilities.emitted_metrics
        universal = universal_validation_keys(DiffusionTrainingStrategy)
        names = [
            f"val_{base}{suffix}"
            for base in ("acquired_bin_nrmse", "unacquired_bin_nrmse")
            for suffix in ("", "_2x", "_8x", "_32x", "_mean")
        ]
        assert (
            unresolved_metric_names(
                names, MetricsRegistry.is_registered, emitted, universal=universal
            )
            == []
        )
        assert metric_higher_is_better("val_acquired_bin_nrmse_8x") is False
        assert DiffusionTrainingStrategy.routes_context_metric("acquired_bin_nrmse")
        assert DiffusionTrainingStrategy.routes_context_metric("unacquired_bin_nrmse")

    def test_the_multistep_branch_is_wired_to_the_ensemble(self) -> None:
        """Wiring pin (the behaviour is above): the cold branch resolves N, samples the
        stack, and the producer clears the stash before every prediction."""
        src = inspect.getsource(DiffusionTrainingStrategy._generate_validation_prediction)
        assert "self._resolve_validation_ensemble(gen)" in src
        assert "self._sample_ensemble_chunked(" in src
        assert "self._ensemble_members = None" in src
        consumer = inspect.getsource(DiffusionTrainingStrategy._compute_validation_metrics)
        assert consumer.index("self._ensemble_members = None") < consumer.index(
            "self._ensemble_metrics("
        )

    def test_the_emitted_keys_resolve_through_the_witness_and_the_mixin(self) -> None:
        import spectramr.core.metrics  # noqa: F401  -- populate the registry
        from spectramr.core.metrics.registry import MetricsRegistry
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            universal_validation_keys,
        )
        from spectramr.infrastructure.validation.witness.checks.validation_metric_checks import (
            unresolved_metric_names,
        )

        emitted = DiffusionTrainingStrategy.capabilities.emitted_metrics
        universal = universal_validation_keys(DiffusionTrainingStrategy)
        names = [
            "empirical_coverage",
            "ensemble_std_mean",
            "val_empirical_coverage_8x",
            "val_empirical_coverage_mean",
            "val_ensemble_std_mean_32x",
            "val_ensemble_std_mean_mean",
        ]
        assert (
            unresolved_metric_names(
                names, MetricsRegistry.is_registered, emitted, universal=universal
            )
            == []
        )
        # The mixin's exact-match rule, which the declaration flipped to strict.
        host = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
        assert host._registry_backed_validation_metrics(
            ["psnr", "ensemble_std_mean", "val_empirical_coverage_mean"]
        ) == ["psnr"]
        with pytest.raises(ValueError, match="neither registered nor emitted"):
            host._registry_backed_validation_metrics(["val_nmse"])


# ===========================================================================
# The strategy may not keep its own list of what the arm optimizes
# (#1918, #1919)
# ===========================================================================

_LOSS_OWNING_METHODS = ("_setup_strategy_specific_components", "_compute_losses_impl")
# `diffusion.py` defines TWO strategy classes and both override both methods
# (`XDiffusionTrainingStrategy` at :7658 -- setup :7867, step :8014). The claim
# below is about the FILE, so the detector ranges over both; scoped to
# `DiffusionTrainingStrategy` alone it could not see a hand-written list appear
# in the sibling, which is the blindness shape non-negotiable 15 exists to catch.
_LOSS_OWNING_CLASSES = ("DiffusionTrainingStrategy", "XDiffusionTrainingStrategy")
_LOSS_OWNING_SITES = [(c, m) for c in _LOSS_OWNING_CLASSES for m in _LOSS_OWNING_METHODS]


def _loss_site_tree(name: str, cls_name: str = "DiffusionTrainingStrategy"):
    """Parse one loss-owning method.

    NOT named ``_method_tree``: this module already defines that at module scope
    for the validation-metric tests, with a different parameter (a method object,
    not a name) and a different return type (``FunctionDef``, not ``Module``). A
    second ``def`` of the same name is a silent rebinding -- the later one wins
    for every caller in the file, including the eight that came first.
    """
    from spectramr.infrastructure.training.strategies import diffusion

    cls = getattr(diffusion, cls_name)
    # An inherited method would hand back `base.py`'s source under the subclass's
    # name -- auditing the same bytes twice and counting it as coverage.
    assert name in vars(cls), f"{cls_name} does not define {name}; the site moved"
    return ast.parse(textwrap.dedent(inspect.getsource(getattr(cls, name))))


def _named_losses(tree) -> set[str]:
    """Registered canonical loss names appearing as string constants.

    Keyed on the registry rather than "any string", so the surviving
    `["g_total_loss", "loss", "diffusion"]` metric filter — none of which is a
    registered loss — does not read as a violation.
    """
    from spectramr.models.losses.registry import LossRegistry

    known = set(LossRegistry.list_available())
    return {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in known
    }


@pytest.mark.unit
class TestNoSecondDeclarationSurface:
    """Three hand-written lists claimed to know the objective and each read one
    schema block, so all three were empty for the 58 `kspace_filling` arms that
    declare theirs in the domain lists. The weight table is the owner now
    (non-negotiable 17); a list here is a fourth owner reappearing."""

    @pytest.mark.parametrize(("cls_name", "method"), _LOSS_OWNING_SITES)
    def test_the_method_names_no_loss(self, cls_name: str, method: str) -> None:
        named = _named_losses(_loss_site_tree(method, cls_name))
        assert named == set(), (
            f"{cls_name}.{method} names losses directly: {sorted(named)}. Read "
            "`self._declared_loss_keys` / `self._weight_table` instead — a list "
            "here mirrors one schema block and silently misses the rest."
        )

    @pytest.mark.parametrize(("cls_name", "method"), _LOSS_OWNING_SITES)
    def test_the_method_holds_no_enable_weight_triples(self, cls_name: str, method: str) -> None:
        """The second shape the same defect took: `[(name, enabled, weight), ...]`.

        Caught separately because a triple list can be built from `getattr`
        lookups with no loss name spelled as a constant at all.
        """
        tree = _loss_site_tree(method, cls_name)
        triples = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.List)
            and sum(isinstance(e, ast.Tuple) and len(e.elts) == 3 for e in node.elts) >= 2
        ]
        assert not triples, (
            f"{cls_name}.{method} rebuilds an (name, enabled, weight) table; "
            "`build_loss_weight_table` already resolved exactly that."
        )

    def test_this_module_binds_no_helper_name_twice(self) -> None:
        """A second module-level ``def`` of a live name is a silent rebinding.

        This file already had ``_method_tree`` (a method object -> FunctionDef)
        when these tests were written; adding a same-named helper taking a
        string re-pointed all eight earlier call sites at the new one. Nothing
        raised at import, ruff saw nothing, and the tests added here still
        passed -- only the tests that were NOT re-run went red. Checked here
        because this module is where the collision happened.
        """
        import collections

        tree = ast.parse(Path(__file__).read_text())
        names = [
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        dupes = {n: c for n, c in collections.Counter(names).items() if c > 1}
        assert not dupes, (
            f"module-level names bound more than once: {dupes}. The last "
            "definition wins for every caller in the file, including earlier ones."
        )

    def test_the_registry_oracle_is_not_vacuous(self) -> None:
        """If the loss registry were empty the two tests above would pass on any
        code at all."""
        from spectramr.models.losses.registry import LossRegistry

        known = set(LossRegistry.list_available())
        assert len(known) > 100
        assert {"complex_l1", "log_spectral", "hfen", "l1", "l2"} <= known


@pytest.mark.unit
class TestTheProductionPathCallsTheSSOT:
    """Non-negotiable 16: registering the helper is the easy half."""

    def test_setup_emits_the_banner(self) -> None:
        tree = _loss_site_tree("_setup_strategy_specific_components")
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "_log_loss_objective" in calls

    def test_the_step_fills_from_the_declared_set(self) -> None:
        tree = _loss_site_tree("_compute_losses_impl")
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "_ensure_declared_losses_present" in calls

    def test_the_banner_actually_reaches_the_logger(self) -> None:
        """Observed, not inferred: drive the real method and read the spy."""
        from types import SimpleNamespace

        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

        arm = _KSPACE_ARM_DIR / "experiment_11_sense_bridge_critic.yaml"
        assert arm.exists(), f"missing experiment config: {arm}"

        strategy = BaseTrainingStrategy.__new__(BaseTrainingStrategy)
        strategy.config = TrainingSettings.from_yaml(str(arm))
        lines: list[str] = []
        strategy.logging_service = SimpleNamespace(log_info=lines.append, log_warning=lines.append)

        strategy._log_loss_objective("[DiffusionStrategy]")

        assert lines, "the banner printed nothing — the state this work removed"
        assert lines[0].startswith("[DiffusionStrategy] Configured Losses (")
        assert not lines[0].startswith("[DiffusionStrategy] Configured Losses (0)"), (
            "this arm trains seven losses; a zero count is the original bug"
        )
        assert any("adversarial" in ln and "0.0100" in ln for ln in lines)


_KSPACE_ARM_DIR = (
    Path(__file__).resolve().parents[5] / "experiments" / "inprogress" / "kspace_filling"
)
_KSPACE_ARMS = sorted(p.relative_to(_KSPACE_ARM_DIR) for p in _KSPACE_ARM_DIR.rglob("*.yaml"))


@pytest.mark.unit
class TestFillSetMatchesTheCsvHeader:
    """The zero-fill exists so a declared loss gets a CSV column. If the two
    sets disagree the fix is worse than the bug: a key the header lacks is
    dropped by the writer (`extrasaction="ignore"`) and reported missing on
    every single step."""

    def test_the_cohort_is_not_empty(self) -> None:
        assert len(_KSPACE_ARMS) > 50, f"corpus not found under {_KSPACE_ARM_DIR}"

    @pytest.mark.parametrize("arm", [str(a) for a in _KSPACE_ARMS])
    def test_declared_keys_equal_the_header_keys(self, arm: str) -> None:
        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.training.builders.loss_builder import (
            STRATEGY_MANAGED_LOSSES,
        )
        from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
        from spectramr.models.losses.weights import canonical_loss_name

        strategy = BaseTrainingStrategy.__new__(BaseTrainingStrategy)
        strategy.config = TrainingSettings.from_yaml(str(_KSPACE_ARM_DIR / arm))

        # The CSV header's own owner: `training_loop` builds it from this call.
        header = {
            canonical_loss_name(name)
            for name, weight in strategy.config.losses.get_enabled_losses().items()
            if weight > 0
        } - STRATEGY_MANAGED_LOSSES

        declared = strategy._declared_loss_keys
        assert not (header - declared), (
            f"header promises columns the fill set misses: {sorted(header - declared)}"
        )
        assert not (declared - header), (
            f"fill set writes keys the header lacks (dropped, then reported "
            f"missing every step): {sorted(declared - header)}"
        )


# ---------------------------------------------------------------------------
# Non-Cartesian sample-domain fidelity: the measurement's route to the losses
# ---------------------------------------------------------------------------
# ``q_sample`` grids the off-grid samples away, so the fidelity term cannot
# recover them from ``pred``/``target``. These pin the out-of-band route --
# ``SimpleNamespace`` rather than ``MagicMock`` throughout, because a mock
# auto-creates ``kspace_process.last_sample_measurement`` as a truthy child and
# every assertion below would pass against a generator that publishes nothing.


def test_sample_measurement_is_none_on_a_cartesian_arm() -> None:
    """The grid process has no sample axis, and absent is the correct answer."""
    strategy = SimpleNamespace(generator_model=SimpleNamespace(kspace_process=None))
    assert DiffusionTrainingStrategy._sample_measurement_for_loss(strategy) is None


def test_sample_measurement_reads_the_process_stash() -> None:
    """The strategy reads the measurement the forward process published."""
    measurement = object()
    strategy = SimpleNamespace(
        generator_model=SimpleNamespace(
            kspace_process=SimpleNamespace(last_sample_measurement=measurement)
        )
    )
    assert DiffusionTrainingStrategy._sample_measurement_for_loss(strategy) is measurement


def test_sample_measurement_unwraps_the_ddp_wrapper() -> None:
    """A DDP-wrapped generator hides the process one attribute deeper."""
    measurement = object()
    inner = SimpleNamespace(
        kspace_process=SimpleNamespace(last_sample_measurement=measurement)
    )
    strategy = SimpleNamespace(generator_model=SimpleNamespace(module=inner))
    assert DiffusionTrainingStrategy._sample_measurement_for_loss(strategy) is measurement


def test_the_loss_computer_call_forwards_the_measurement() -> None:
    """PLANTED VIOLATION: a stash nothing forwards is the facade shape.

    The helper above can be correct while the measurement never reaches a loss.
    Read the real call site rather than trusting that it was edited.
    """
    source = inspect.getsource(DiffusionTrainingStrategy._compute_losses_impl)
    call = next(
        node
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compute"
    )
    assert "sample_measurement" in {kw.arg for kw in call.keywords}, (
        "the loss computer call must forward the rung measurement"
    )


def test_a_late_top_rung_is_warned_about_once_and_draws_match_it() -> None:
    """#2066: the kspace_filling curriculum opens t=28 at iteration 56000 of 70000.

    The warning is the observable half; the draw at iteration 55999 is the proof
    that the number it states is the sampler's own cap.
    """
    strategy = _timestep_strategy(
        "uniform", floor=0, max_iterations=70_000, num_timesteps=29,
        curriculum_start=1, curriculum_rate=0.0005,
    )
    strategy.config.training.diffusion = SimpleNamespace(timesteps=29)
    state = strategy._resolve_curriculum_once()
    assert state.top_opens_at == 56_000
    warnings = [c.args[0] for c in strategy.logging_service.log_warning.call_args_list]
    assert len(warnings) == 1 and "iteration 56000 of 70000" in warnings[0]
    before = DiffusionTrainingStrategy.sample_timesteps(strategy, batch_size=4096, iteration=55_999)
    after = DiffusionTrainingStrategy.sample_timesteps(strategy, batch_size=4096, iteration=56_000)
    assert before.max().item() < 28 == after.max().item()


# ---------------------------------------------------------------------------
# Cold-diffusion validation S-maps and chunking (#1914).
# ---------------------------------------------------------------------------


def test_cold_validation_hands_each_chunk_its_own_slice_of_this_batchs_smaps(
    monkeypatch,
) -> None:
    """Single-step cold validation estimated the batch's maps, stored them on the
    strategy and concatenated them into the input, but never passed them to the
    generator, whose SENSE projection then used the maps cached on the last
    training forward. And ``_forward_chunked`` sliced only the input and the
    timesteps, so chunk 2 of a split batch met full-batch masks and measurements.
    """
    from spectramr.infrastructure.training.strategies import diffusion as diffusion_mod

    batch, chunk = 4, 2
    maps = torch.randn(batch, 1, 8, 8, dtype=torch.complex64)
    monkeypatch.setattr(diffusion_mod, "estimate_smaps_calibrated", lambda *a, **k: maps)

    mock = _validation_strategy(chunk_size=chunk)
    mock._is_cold_diffusion.return_value = True
    mock.config.model = SimpleNamespace(model_type="kspace_cold_diffusion")
    mock.config.validation = SimpleNamespace(
        loader=SimpleNamespace(chunk_size=chunk),
        sampling=SimpleNamespace(enable_multistep_cold=False),
    )
    mask = torch.ones(batch, 1, 8, 8)
    mock.generate_and_process_mask.return_value = mask
    mock._apply_masking_and_verify.side_effect = lambda x, m, **_: x * m

    seen: list[dict] = []

    def _record(model_input, timesteps=None, is_latent_diffusion=False, gen_kwargs=None):
        seen.append({"n": model_input.shape[0], **gen_kwargs})
        return torch.zeros(model_input.shape[0], 2, 8, 8)

    mock._forward_through_model.side_effect = _record
    inp = torch.randn(batch, 2, 8, 8)
    DiffusionTrainingStrategy._generate_validation_prediction(
        mock,
        input_batch=inp,
        target_batch=inp.clone(),
        timestep=torch.ones(batch, dtype=torch.long),
        batch_data={},
        kwargs={},
    )
    swallowed = [
        c.args[0]
        for c in mock.logging_service.log_warning.call_args_list
        if c.args and "Validation generation failed" in str(c.args[0])
    ]
    assert not swallowed, swallowed
    assert [c["n"] for c in seen] == [chunk, chunk]
    for i, call in enumerate(seen):
        rows = slice(i * chunk, (i + 1) * chunk)
        assert torch.equal(call["smaps"], maps[rows])
        assert call["mask"].shape[0] == chunk
        assert call["kspace_measured"].shape[0] == chunk


def test_x_diffusion_without_a_horizon_raises_rather_than_sampling_1000_steps():
    """PLANTED (#2443): a missing training.diffusion fell through ``... or 1000``."""
    from spectramr.infrastructure.training.strategies.diffusion import XDiffusionTrainingStrategy

    strategy = object.__new__(XDiffusionTrainingStrategy)
    strategy.env = SimpleNamespace(generator=torch.nn.Identity(), losses={})
    strategy.config = SimpleNamespace(
        training=SimpleNamespace(diffusion=None),
        model=SimpleNamespace(model_type="x_diffusion"),
    )
    strategy.logging_service = SimpleNamespace(log_info=lambda *a, **k: None)
    strategy._loss_dict_reuse = {}
    x = torch.randn(1, 1, 8, 8)
    with pytest.raises(ValueError, match=r"training\.diffusion\.timesteps"):
        strategy._compute_losses_impl(input_batch=x, target_batch=x, epoch=0)


def test_the_clamp_statistics_reach_the_training_metrics_as_device_tensors():
    """Wiring: the generator's last clip stats land in the loss dict without a host sync."""
    import inspect

    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    src = inspect.getsource(DiffusionTrainingStrategy._compute_losses_impl)
    assert '"clip_bind_frac"] = clip_stats[0]' in src
    assert '"clip_overshoot_mean"] = clip_stats[1]' in src
    assert "float(clip_stats" not in src and ".item()" not in src.split("clip_stats = ")[1][:400]


def test_the_output_snapshot_also_fires_at_each_validation_step():
    """The early cadence spends its budget mid-curriculum; a second tag covers validation."""
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    s = object.__new__(DiffusionTrainingStrategy)
    s._cached_log_interval = 5000
    s._cached_val_interval = 10000
    due = DiffusionTrainingStrategy._output_snapshot_tags_due
    assert due(s, 500) == ("model_output_dc",)
    assert due(s, 10000) == ("model_output_dc", "model_output_dc_validation")
    assert due(s, 10250) == ()
    assert due(s, 0) == ("model_output_dc",)


def test_no_validation_tag_when_validation_is_off():
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    s = object.__new__(DiffusionTrainingStrategy)
    s._cached_log_interval = 5000
    s._cached_val_interval = 0
    assert DiffusionTrainingStrategy._output_snapshot_tags_due(s, 10000) == ("model_output_dc",)


def test_the_t0_probe_drops_the_metrics_a_fully_sampled_mask_leaves_undefined():
    """At t=0 there is no unmeasured band: those columns were nan in every row of 59 runs."""
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
        drop_unmeasured_band_metrics,
    )

    names = DiffusionTrainingStrategy._MEASUREMENT_AWARE_METRICS
    metrics = {
        "val_psnr": 30.0,
        "val_hfen": 0.4,
        "val_zf_psnr": 25.0,
        "val_eta_null": float("nan"),
        "val_null_band_energy_deficit": float("nan"),
        "val_zf_fabrication_excess": float("nan"),
        "val_zf_delta_null_band_nrmse_outer": float("nan"),
    }
    kept = drop_unmeasured_band_metrics(metrics, names)
    assert kept == {"val_psnr": 30.0, "val_hfen": 0.4, "val_zf_psnr": 25.0}


def test_the_t0_probe_keeps_the_acquired_bin_error_and_drops_the_rest():
    """The acquired-bin error is defined under the probe's all-ones mask; the unmeasured-band
    metrics, unacquired_bin_nrmse included, are not and stay dropped."""
    import inspect

    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
        drop_unmeasured_band_metrics,
    )

    dropped = DiffusionTrainingStrategy._t0_probe_dropped_metrics()
    metrics = {
        "val_acquired_bin_nrmse": 0.12,
        "val_zf_acquired_bin_nrmse": 0.30,
        "val_unacquired_bin_nrmse": 0.0,
        "val_eta_null": float("nan"),
    }
    kept = drop_unmeasured_band_metrics(metrics, dropped)
    assert kept == {"val_acquired_bin_nrmse": 0.12, "val_zf_acquired_bin_nrmse": 0.30}
    # Planted: the full measurement-aware set (the old call) drops it.
    assert "val_acquired_bin_nrmse" not in drop_unmeasured_band_metrics(
        metrics, DiffusionTrainingStrategy._MEASUREMENT_AWARE_METRICS
    )
    assert "self._t0_probe_dropped_metrics()" in inspect.getsource(
        DiffusionTrainingStrategy._t0_pre_dc_probe_metrics
    )


def _with_generator(gen):
    """A strategy class whose ``generator_model`` property returns ``gen``."""
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    return type("_Strategy", (DiffusionTrainingStrategy,), {"generator_model": property(lambda self: gen)})


def test_the_output_snapshot_carries_the_backbone_tap_and_the_image_domain_maps():
    """#2488: the pre-DC tensor already includes the coil projection, so the
    backbone's own output is snapshotted beside it, and the maps go under
    ``coil_maps`` where the per-coil grid renders them."""
    from types import SimpleNamespace

    backbone = torch.rand(1, 8, 16, 16)
    s = object.__new__(_with_generator(SimpleNamespace(last_backbone_output=backbone)))
    s._current_smaps = torch.ones(1, 4, 16, 16, dtype=torch.complex64) / 2
    snap = s._build_output_snapshot(
        torch.rand(1, 8, 16, 16),
        (torch.rand(1, 8, 16, 16), torch.rand(1, 8, 16, 16)),
        torch.rand(1, 8, 16, 16),
        torch.rand(1, 8, 16, 16),
        None,
    )
    assert snap["model_output_backbone"] is backbone
    assert "model_output_pre_dc" in snap and "model_output_post_dc" in snap
    assert torch.equal(snap["coil_maps"], s._current_smaps)
    assert "smaps" not in snap


def test_the_output_snapshot_carries_the_projected_backbone_beside_the_raw_one():
    from types import SimpleNamespace

    raw, projected = torch.rand(1, 8, 16, 16), torch.rand(1, 8, 16, 16)
    gen = SimpleNamespace(last_backbone_output=raw, last_backbone_projected=projected)
    s = object.__new__(_with_generator(gen))
    snap = s._build_output_snapshot(
        torch.rand(1, 8, 16, 16), torch.rand(1, 8, 16, 16), torch.rand(1, 8, 16, 16),
        torch.rand(1, 8, 16, 16), None,
    )
    assert snap["model_output_backbone_projected"] is projected


def test_the_backbone_tap_is_left_out_when_its_shape_does_not_match():
    from types import SimpleNamespace

    s = object.__new__(_with_generator(SimpleNamespace(last_backbone_output=torch.rand(1, 8, 8, 8))))
    snap = s._build_output_snapshot(
        torch.rand(1, 8, 16, 16), torch.rand(1, 8, 16, 16), torch.rand(1, 8, 16, 16),
        torch.rand(1, 8, 16, 16), None,
    )
    assert "model_output_backbone" not in snap


def test_chunked_validation_sums_the_unfitted_envelope_count_across_chunks():
    """Each chunk's sampler record replaces the last; the per-sample count must not."""
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
    from tests.utils.config_block_stub import block_stub

    gen = SimpleNamespace(last_reverse_stats=None)

    def sample(measurement, mask, inference_timesteps, smaps=None, start_timestep=None):
        # A fresh record per call, like the generator's per-call sampler.
        gen.last_reverse_stats = {"decay_unfitted": int(measurement.flatten()[0] < 0)}
        return measurement

    gen.sample = sample
    strat = SimpleNamespace(
        config=SimpleNamespace(validation=block_stub("validation", val_chunk_size=1))
    )
    meas = torch.tensor([-1.0, 1.0, -1.0]).view(3, 1, 1, 1).expand(3, 2, 4, 4).contiguous()
    DiffusionTrainingStrategy._sample_multistep_chunked(
        strat, gen, meas, torch.ones(3, 1, 4, 4), steps=4
    )
    assert gen.last_reverse_stats["decay_unfitted"] == 2


def test_a_generation_error_propagates_instead_of_returning_none() -> None:
    """#1152/#181: the blanket except turned a wiring error into ``None``, every
    batch then "succeeded", and the run reported success with no validation
    number. The error must now reach ``_run_validation``'s per-batch accounting.
    """
    mock = _validation_strategy()

    def _planted(*_a, **_k):
        raise TypeError("planted: kwargs no registered accelerator reads")

    mock._contrast_idx_from_batch = _planted
    with pytest.raises(TypeError, match="planted") as info:
        DiffusionTrainingStrategy._generate_validation_prediction(
            mock,
            input_batch=torch.randn(1, 1, 8, 8),
            target_batch=torch.zeros(1, 1, 8, 8),
            timestep=torch.tensor([1]),
            batch_data={},
            kwargs={},
        )
    assert any("Validation generation failed" in note for note in info.value.__notes__)


def test_a_missing_prediction_is_not_scored_as_the_failure_sentinel() -> None:
    """#181(a): ``hr_fakes is None`` returned ``{"validation_error": 1.0}``, a
    finite value that let a run with no prediction report success."""
    src = inspect.getsource(DiffusionTrainingStrategy.validation_step)
    branch = src[src.index("if self.config.undersampling is None:") :]
    branch = branch[: branch.index("single_metrics = self._compute_validation_metrics")]
    assert "raise RuntimeError(" in branch
    assert '"validation_error"' not in branch


def _hfen_loss_and_metric_mock(loss_value: float) -> MagicMock:
    """A host that scores ``hfen`` both as a loss term and as a metric (#2119)."""
    mock = _zf_mock(metrics=("hfen",))
    mock.context = SimpleNamespace(loss_fn=object())
    mock.env.losses = {"hfen": lambda pred, target: torch.tensor(loss_value)}
    mock._get_loss_weight = lambda name, iteration: 1.0
    return mock


def test_a_loss_and_a_metric_of_one_name_keep_separate_columns():
    """The loss stamp no longer shares ``val_hfen`` with the metric that overwrote it."""
    torch.manual_seed(0)
    target = torch.rand(2, 1, 16, 16)
    pred = target + 0.1 * torch.randn_like(target)
    out = _run_validation_metrics(_hfen_loss_and_metric_mock(0.25), pred, target, pred)

    assert float(out[validation_loss_key("hfen")]) == pytest.approx(0.25)
    assert "val_hfen" in out and float(out["val_hfen"]) != pytest.approx(0.25)
    # ``val_loss`` decomposes into the loss columns published beside it.
    assert float(out["val_loss"]) == pytest.approx(float(out[validation_loss_key("hfen")]))


def test_loss_columns_cannot_collide_with_a_registered_metric():
    """Planted: the pre-fix ``val_{loss}`` spelling collides on ``hfen``; the namespace never does."""
    from spectramr.core.metrics.registry import MetricsRegistry
    from spectramr.models.losses.registry import LossRegistry

    metric_columns = {f"val_{name}" for name in MetricsRegistry._metrics}
    losses = LossRegistry.list_available()
    assert "val_hfen" in {f"val_{name}" for name in losses} & metric_columns
    assert not {validation_loss_key(name) for name in losses} & metric_columns


def _cold_mock():
    mock = MagicMock()
    mock.generator_model = _gen(conditional=False)
    mock._current_smaps = None
    del mock.prior_model
    mock.config = SimpleNamespace(undersampling=None)
    return mock


def test_training_hands_the_generator_the_masked_measurement() -> None:
    """The output ceiling is read off ``kspace_measured``; training must not see bins validation hides."""
    from spectramr.models.diffusion.kspace_process import band_local_magnitude_ceiling

    # Magnitude falls with radius from the centre, as k-space does; only the central columns are
    # acquired, so the outer bands are populated in the full tensor and empty in the measurement.
    ky, kx = torch.meshgrid(torch.arange(32) - 16, torch.arange(32) - 16, indexing="ij")
    full = (1.0 / (1.0 + (ky**2 + kx**2).float().sqrt())).expand(2, 8, 32, 32).clone()
    mask = torch.zeros(2, 1, 32, 32)
    mask[..., 12:20] = 1.0
    kw = DiffusionTrainingStrategy._build_generator_kwargs(
        _cold_mock(),
        is_cold_diffusion=True,
        is_latent_diffusion=False,
        input_batch=full,
        target_batch=full,
        batch_data=None,
        mask=mask,
    )
    measured = kw["kspace_measured"]
    assert torch.equal(measured, full * mask)
    # The validation path's measurement gives the same ceiling; the unmasked one does not.
    val_ceiling = band_local_magnitude_ceiling(full * mask, 1.3, log_scaled=False)
    assert torch.equal(band_local_magnitude_ceiling(measured, 1.3, log_scaled=False), val_ceiling)
    assert not torch.equal(band_local_magnitude_ceiling(full, 1.3, log_scaled=False), val_ceiling)


# --- Padded lines are not "unsampled" (#2160) --------------------------------
# M4Raw writes 195 (T1/T2) or 198 (FLAIR) of 256 phase-encode lines; the rest
# are exact zeros in input and target, and the rung mask is built over ACQUIRED
# lines only, so they always read mask == 0. Counting them as unsampled bins
# dilutes the mean of the pre-DC null-band term.


def _padded_pre_dc_case(n_pad: int):
    """A 16-line plane plus ``n_pad`` zero columns; kept lines are every 2nd of the 16."""
    from spectramr.models.diffusion.acquired_line_budget import lines_to_plane

    torch.manual_seed(0)
    n = 16 + n_pad
    pre = torch.zeros(1, 2, 8, n)
    tgt = torch.zeros(1, 2, 8, n)
    pre[..., :16] = torch.randn(1, 2, 8, 16)
    tgt[..., :16] = torch.randn(1, 2, 8, 16)
    kept = torch.zeros(1, n, dtype=torch.bool)
    kept[:, 0:16:2] = True
    acquired = torch.zeros(1, n, dtype=torch.bool)
    acquired[:, :16] = True
    return (
        pre,
        tgt,
        lines_to_plane(kept, (8, n), -1),
        lines_to_plane(acquired, (8, n), -1),
    )


def _pre_dc_term(pre, tgt, mask, acquired) -> torch.Tensor:
    strategy = MagicMock(spec=DiffusionTrainingStrategy)
    strategy._loss_dict_reuse = {}
    strategy.config = SimpleNamespace(
        losses=SimpleNamespace(reconstruction=SimpleNamespace(lambda_pre_dc_kspace=1.0))
    )
    strategy._unsampled_weight = DiffusionTrainingStrategy._unsampled_weight
    return DiffusionTrainingStrategy._add_pre_dc_fidelity(
        strategy, torch.zeros(()), (pre, pre), tgt, mask, acquired
    )


def test_pre_dc_term_ignores_padded_lines_when_acquired_is_given() -> None:
    base = _pre_dc_term(*_padded_pre_dc_case(0)[:3], None)
    for n_pad in (4, 12):
        pre, tgt, mask, acq = _padded_pre_dc_case(n_pad)
        assert torch.allclose(_pre_dc_term(pre, tgt, mask, acq), base)
        # Planted violation: the legacy weight DOES move with the padding.
        assert not torch.allclose(_pre_dc_term(pre, tgt, mask, None), base)


def test_pre_dc_term_without_acquired_is_the_legacy_value() -> None:
    pre, tgt, mask, _ = _padded_pre_dc_case(6)
    w = (1.0 - mask).expand_as(pre)
    expected = (w * (pre - tgt).abs()).sum() / w.sum()
    assert torch.allclose(_pre_dc_term(pre, tgt, mask, None), expected)


def test_unsampled_weight_counts_only_acquired_and_unkept_bins() -> None:
    pre, _, mask, acq = _padded_pre_dc_case(5)
    w =DiffusionTrainingStrategy._unsampled_weight(mask, pre, acq)
    assert w is not None
    # 8 unkept acquired lines x 8 rows; the 5 padded columns carry no weight.
    assert float(w.sum()) == 8 * 8
    assert float(w[..., 16:].sum()) == 0.0


def test_unsampled_weight_returns_none_when_acquired_lines_are_all_kept() -> None:
    """Identity rung on padded data: the helper reports no unmeasured bins (the caller decides)."""
    pre, _, _, acq = _padded_pre_dc_case(5)
    full = acq.clone()
    assert DiffusionTrainingStrategy._unsampled_weight(full, pre, acq) is None


def _pre_dc_terms_at_identity(lam_acquired: float) -> dict:
    """Both pre-DC terms on a padded plane whose every acquirable line is kept (t=0)."""
    pre, tgt, _, acq = _padded_pre_dc_case(5)
    strategy = MagicMock(spec=DiffusionTrainingStrategy)
    strategy._loss_dict_reuse = {}
    strategy.config = SimpleNamespace(
        losses=SimpleNamespace(
            reconstruction=SimpleNamespace(
                lambda_pre_dc_kspace=1.0, lambda_pre_dc_acquired=lam_acquired
            )
        )
    )
    strategy._unsampled_weight = DiffusionTrainingStrategy._unsampled_weight
    strategy._acquired_weight = DiffusionTrainingStrategy._acquired_weight
    DiffusionTrainingStrategy._add_pre_dc_fidelity(
        strategy, torch.zeros(()), (pre + 1.0, pre + 1.0), tgt, acq.clone(), acq
    )
    return strategy._loss_dict_reuse


def test_identity_rung_does_not_score_acquired_bins_twice_when_the_acquired_term_is_on() -> None:
    """The acquired-band term scores every bin at t=0; the null-band fallback must not repeat it."""
    terms = _pre_dc_terms_at_identity(0.5)
    assert float(terms["pre_dc_kspace_l1"]) == 0.0
    assert float(terms["pre_dc_acquired_l1"]) > 0.0


def test_identity_rung_keeps_the_uniform_fallback_when_the_acquired_term_is_off() -> None:
    """Without the acquired-band term the fallback is the only t=0 gradient (planted contrast)."""
    terms = _pre_dc_terms_at_identity(0.0)
    assert float(terms["pre_dc_kspace_l1"]) > 0.0


def test_acquired_plane_of_lifts_the_process_lines_and_is_none_without_a_line_axis() -> None:
    kspace = torch.zeros(2, 2, 4, 6)
    kspace[..., :4] = 1.0  # columns 4-5 are padding
    process = SimpleNamespace(
        line_dim=-1, acquired_lines=lambda k: (k.abs().sum((1, 2)) > 0)  # [B, W]
    )
    strategy = MagicMock(spec=DiffusionTrainingStrategy)
    strategy.generator_model = SimpleNamespace(kspace_process=process)
    plane = DiffusionTrainingStrategy._acquired_plane_of(strategy, kspace)
    assert plane is not None and plane.shape == (2, 1, 4, 6)
    assert float(plane[..., :4].min()) == 1.0 and float(plane[..., 4:].max()) == 0.0

    process.acquired_lines = lambda k: None
    assert DiffusionTrainingStrategy._acquired_plane_of(strategy, kspace) is None


@pytest.mark.parametrize("container", ["training_batch", "dict"])
def test_cold_validation_calibrates_coil_maps_from_the_target(monkeypatch, container) -> None:
    """Validation's maps come from ``target_batch`` whatever ``batch_data`` carries.

    Production hands a ``TrainingBatch``, which the old ``isinstance(batch_data,
    dict)`` read of a ``kspace`` alias never matched; the target was what every run
    calibrated from. PLANTED: the deleted branch would calibrate the ``dict`` case
    from the alias instead.
    """
    from spectramr.data.batch_types import TrainingBatch
    from spectramr.infrastructure.training.strategies import diffusion as diffusion_mod

    batch = 2
    seen: list[torch.Tensor] = []

    def _estimate(acs, **kwargs):
        seen.append(acs)
        return torch.ones(batch, 1, 8, 8, dtype=torch.complex64)

    monkeypatch.setattr(diffusion_mod, "estimate_smaps_calibrated", _estimate)
    mock = _validation_strategy(chunk_size=batch)
    mock._is_cold_diffusion.return_value = True
    mock.config.model = SimpleNamespace(model_type="kspace_cold_diffusion")
    mock.config.validation = SimpleNamespace(
        loader=SimpleNamespace(chunk_size=batch),
        sampling=SimpleNamespace(enable_multistep_cold=False),
    )
    mock.generate_and_process_mask.return_value = torch.ones(batch, 1, 8, 8)
    mock._apply_masking_and_verify.side_effect = lambda x, m, **_: x * m
    mock._forward_through_model.side_effect = lambda model_input, **k: torch.zeros(
        model_input.shape[0], 2, 8, 8
    )
    target = torch.randn(batch, 2, 8, 8)
    alias = torch.randn(batch, 2, 8, 8)
    if container == "dict":
        batch_data = {"kspace": alias}
    else:
        batch_data = TrainingBatch(input=target, target=target, metadata={"kspace": alias})
    DiffusionTrainingStrategy._generate_validation_prediction(
        mock,
        input_batch=target.clone(),
        target_batch=target,
        timestep=torch.ones(batch, dtype=torch.long),
        batch_data=batch_data,
        kwargs={},
    )
    (acs,) = seen
    assert torch.equal(acs, torch.complex(target[:, :1], target[:, 1:]))


class TestValidationLossesScoreTrainingUnits:
    """``val_loss`` is scored on the normalised, log-compressed tensors.

    The k-space losses are built with ``log_scaled=True`` and decompress inside
    their own Fourier bridge. Handing them the physical-unit tensors that the
    metrics consume applied ``expm1`` a second time, so magnitudes saturated at
    ``expm1(DECOMPRESS_MAGNITUDE_CEILING)`` and ``coil_subspace_residual`` (an
    energy) reported ~1e23 on every validation event.
    """

    @staticmethod
    def _method_tree() -> ast.FunctionDef:
        src = textwrap.dedent(
            inspect.getsource(DiffusionTrainingStrategy._compute_validation_metrics)
        )
        return ast.parse(src).body[0]

    def test_loss_operands_are_bound_before_decompression(self) -> None:
        fn = self._method_tree()
        bind_line = decompress_line = None
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Tuple) and [getattr(e, "id", None) for e in t.elts]
                == ["loss_pred", "loss_target"]
                for t in node.targets
            ):
                bind_line = node.lineno
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "decompress_scored_against"
            ):
                decompress_line = node.lineno
        assert bind_line is not None, "loss operands are no longer bound"
        assert decompress_line is not None
        assert bind_line < decompress_line, "loss operands bound after decompression"

    def test_every_loss_call_uses_the_training_unit_operands(self) -> None:
        fn = self._method_tree()
        calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_call_safe_loss"
        ]
        assert calls, "validation no longer scores the declared losses"
        for call in calls:
            names = [getattr(a, "id", None) for a in call.args[1:3]]
            assert names == ["loss_pred", "loss_target"], names

    def test_double_decompression_is_what_the_binding_prevents(self) -> None:
        """Planted shape: the bridge applied to an already-decompressed tensor."""
        from spectramr.models.losses.physics_losses import kspace_to_image

        torch.manual_seed(0)
        raw = torch.randn(1, 2, 16, 16, dtype=torch.complex64) * 50.0
        compressed = compress_kspace_log(raw)
        once = kspace_to_image(compressed, log_scaled=True)
        twice = kspace_to_image(decompress_kspace_log(compressed), log_scaled=True)
        assert torch.allclose(once, kspace_to_image(raw), rtol=1e-3, atol=1e-3)
        assert twice.max() > 1e6 * once.max()


# ── Pattern-C acquisition table binding (LM-20) ──────────────────────────────


def _acq_strategy_mock(gen, *, enabled=True, params=None):
    from spectramr.config.schemas.data import AcquisitionParamsSchema

    rows = params if params is not None else [
        AcquisitionParamsSchema(name="T1w", TE=12.0, TR=600.0, FA=70.0),
        AcquisitionParamsSchema(name="T2w", TE=80.0, TR=4000.0),
        AcquisitionParamsSchema(name="FLAIR", TE=120.0, TR=8000.0, TI=2200.0),
    ]
    mock = MagicMock()
    mock.generator_model = gen
    mock.config.data.multi_contrast = SimpleNamespace(
        enabled=enabled, acquisition_params=rows, contrast_map={"T1w": 0, "T2w": 1, "FLAIR": 2}
    )
    return mock


def _film_ldm(film: bool):
    return _real_ldm(
        conditional_translation=True,
        use_contrast_guidance=True,
        num_contrasts=3,
        acquisition_film=film,
    )


def test_bind_acquisition_table_reaches_the_generator() -> None:
    gen = _film_ldm(True)
    DiffusionTrainingStrategy._bind_acquisition_table(_acq_strategy_mock(gen))
    assert bool(gen.acquisition_film.table_bound)
    assert gen.acquisition_film.table[2].tolist() == [120.0, 8000.0, 2200.0, 90.0]


def test_bind_acquisition_table_without_params_raises() -> None:
    with pytest.raises(ValueError, match="acquisition_params is empty"):
        DiffusionTrainingStrategy._bind_acquisition_table(_acq_strategy_mock(_film_ldm(True), params=[]))


def test_bind_acquisition_table_is_a_noop_when_film_is_off() -> None:
    mock = _acq_strategy_mock(_film_ldm(False))
    DiffusionTrainingStrategy._bind_acquisition_table(mock)
    mock.logging_service.log_info.assert_not_called()


def test_training_metrics_score_decompressed_denormalised_kspace(monkeypatch) -> None:
    """train_* graded the SENSE image of log-compressed k-space; it must match val_* units."""
    from types import SimpleNamespace

    import torch

    import spectramr.infrastructure.training.utils.kspace_view as kv
    from spectramr.data.transforms.normalization import compress_kspace_log
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    monkeypatch.setattr(kv, "log_scaling_enabled", lambda _cfg: True)
    torch.manual_seed(0)
    target = torch.randn(2, 8, 16, 16) * 3
    pred = target + 0.1 * torch.randn_like(target)
    scale = torch.tensor([2.0, 5.0])
    stub = SimpleNamespace(
        _is_cold_diffusion=lambda: True,
        _train_metric_scale=scale,
        config=SimpleNamespace(
            data=SimpleNamespace(processing=SimpleNamespace(enable_kspace_normalization=True))
        ),
    )
    got_pred, got_target = DiffusionTrainingStrategy._training_metric_inputs(
        stub, compress_kspace_log(pred, channel_dim=1), compress_kspace_log(target, channel_dim=1)
    )
    s = scale.view(-1, 1, 1, 1)
    torch.testing.assert_close(got_target, target * s, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got_pred, pred * s, rtol=1e-4, atol=1e-4)


# ── PR4: rung and timestep provenance on the output snapshots ─────────────────


def _bare_diffusion_for_snapshots(tmp_path=None, *, enabled: bool = True):
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    s = object.__new__(DiffusionTrainingStrategy)
    s.env = SimpleNamespace(run_output_dir=str(tmp_path) if tmp_path else "/nonexistent", ema=object())
    s.config = SimpleNamespace(
        logging=SimpleNamespace(
            snapshots=SimpleNamespace(
                enabled=enabled, max_calls=8, save_images=False, save_json=True, interval_steps=0
            )
        ),
        training=SimpleNamespace(output_dir=str(tmp_path) if tmp_path else "/nonexistent"),
        validation=SimpleNamespace(sampling=SimpleNamespace(enable_multistep_cold=True)),
    )
    s._zf_measurement = torch.full((1, 1, 4, 8), 0.5)
    s._rung_mask = torch.ones(1, 1, 4, 8)
    return s


def test_realised_acceleration_is_per_sample_over_sampled_pe_lines():
    """N / sampled PE lines per sample; the acquired plane narrows the numerator."""
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    s = object.__new__(DiffusionTrainingStrategy)
    gen = SimpleNamespace(kspace_process=SimpleNamespace(line_dim=-1))
    mask = torch.zeros(2, 1, 4, 8)
    mask[0, :, :, [0, 2, 4, 6]] = 1.0  # 4 of 8 PE columns sampled
    mask[1, :, :, [0, 1]] = 1.0  # 2 of 8 sampled
    with patch.object(DiffusionTrainingStrategy, "generator_model", new_callable=PropertyMock, return_value=gen):
        assert DiffusionTrainingStrategy._snapshot_realised_acceleration(s, mask, None) == [2.0, 4.0]

    plane = torch.zeros(2, 1, 4, 8)
    plane[0, :, :, :6] = 1.0  # sample 0 acquired 6 columns: 6 / 4
    plane[1, :, :, :] = 0.0  # sample 1 acquired nothing: sampled lines are unacquired
    with patch.object(DiffusionTrainingStrategy, "generator_model", new_callable=PropertyMock, return_value=gen):
        got = DiffusionTrainingStrategy._snapshot_realised_acceleration(s, mask, plane)
    assert got == [1.5, 0.0]


def test_realised_acceleration_is_unknown_without_a_line_axis():
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    s = object.__new__(DiffusionTrainingStrategy)
    gen = SimpleNamespace(kspace_process=SimpleNamespace(line_dim=None))
    with patch.object(DiffusionTrainingStrategy, "generator_model", new_callable=PropertyMock, return_value=gen):
        assert DiffusionTrainingStrategy._snapshot_realised_acceleration(s, torch.ones(1, 1, 4, 8), None) is None
        assert DiffusionTrainingStrategy._snapshot_realised_acceleration(s, None, None) is None


def test_the_output_snapshot_extras_are_deferred_and_name_their_rung_and_cadence():
    """Wiring: the model_output_dc emitter passes callables, so no sync runs unless written."""
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    src = inspect.getsource(DiffusionTrainingStrategy._compute_losses_impl)
    assert '"timesteps": lambda: timesteps.detach().flatten().tolist()' in src
    assert "lambda: self._snapshot_realised_acceleration(" in src
    assert '"train_step_at_val_cadence"' in src and '"train_step_at_log_cadence"' in src


def test_cascade_val_8x_is_emitted_for_the_8x_rung_on_the_first_batch_under_val_source():
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    s = _bare_diffusion_for_snapshots()
    s.save_debug_snapshot = MagicMock()
    seen: dict = {}

    def _record(*args, **kwargs):
        seen["phase"] = s._snapshot_phase

    s.save_debug_snapshot.side_effect = _record
    hr_fakes = torch.rand(1, 1, 4, 8)
    target = torch.rand(1, 1, 4, 8)
    gen = SimpleNamespace(_reverse_mode="replace_freeze_dc")
    with (
        patch.object(
            DiffusionTrainingStrategy, "generator_model", new_callable=PropertyMock, return_value=gen
        ),
        s.snapshot_source("val"),
    ):
        s._emit_cascade_val_snapshot(hr_fakes, target, accel=8, timestep=17)

    assert seen["phase"] == "val"
    (tensors,), kwargs = s.save_debug_snapshot.call_args
    assert kwargs["tag"] == "cascade_val_8x"
    assert set(tensors) == {"model_output_post_dc", "target", "input", "mask"}
    # The cascade's own prediction is what is shown: no second sampler call.
    assert tensors["model_output_post_dc"].data_ptr() == hr_fakes.data_ptr()
    assert kwargs["extra"] == {
        "acceleration": 8,
        "timestep": 17,
        "validation_sampling": "multistep_cold",
        "reverse_sampling_mode": "replace_freeze_dc",
        "ema": True,
    }
    assert kwargs["in_kspace_keys"] == {"input", "target", "model_output_post_dc"}


def test_cascade_val_8x_is_gated_to_the_8x_first_batch_before_the_metrics_pop_the_rung():
    """Gate and ordering are wiring: the rung state is read before _compute_validation_metrics pops it."""
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    src = inspect.getsource(DiffusionTrainingStrategy.validation_step)
    assert "if accel == 8 and not _is_heldout and batch_idx == 0:" in src
    assert 'with self.snapshot_source("val"):' in src
    emit = src.index("self._emit_cascade_val_snapshot(")
    assert emit < src.index("level_metrics = self._compute_validation_metrics(")


def test_cascade_val_8x_writes_nothing_when_snapshots_are_disabled(tmp_path):
    """Real writer, disabled config: no directory, and the control run with snapshots on writes one."""
    s_off = _bare_diffusion_for_snapshots(tmp_path / "off", enabled=False)
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    s_off.save_debug_snapshot = BaseTrainingStrategy.save_debug_snapshot.__get__(s_off)
    hr = torch.rand(1, 1, 4, 8)
    s_off._emit_cascade_val_snapshot(hr, torch.rand(1, 1, 4, 8), accel=8, timestep=3)
    assert not list((tmp_path / "off").rglob("cascade_val_8x*"))

    s_on = _bare_diffusion_for_snapshots(tmp_path / "on", enabled=True)
    s_on.save_debug_snapshot = BaseTrainingStrategy.save_debug_snapshot.__get__(s_on)
    s_on._emit_cascade_val_snapshot(hr, torch.rand(1, 1, 4, 8), accel=8, timestep=3)
    assert list((tmp_path / "on").rglob("cascade_val_8x*")), "control run wrote nothing"
