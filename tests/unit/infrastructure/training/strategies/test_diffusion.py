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
from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.config.schemas.enums import PredictionType
from spectramr.data.batch_types import TrainingBatch
from spectramr.data.transforms.normalization import (
    DECOMPRESS_MAGNITUDE_CEILING,
    decompress_kspace_log,
)
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.schedulers.diffusion_scheduler import (
    DiffusionScheduler,
)
from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)
from spectramr.infrastructure.training.utils.kspace_view import decompress_for_view


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
        scale_factor=torch.tensor(1.0),
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


def test_mask_coverage_check_reads_the_real_acceleration_field() -> None:
    """The DEBUG mask-coverage check was gated on
    `hasattr(self.config.data, "acceleration")` — permanently False, since the
    acceleration factor lives on its own block. It now reads
    `acceleration.base_acceleration`, so the check can actually run."""
    import inspect

    from spectramr.config.schemas.acceleration import AccelerationConfigSchema
    from spectramr.config.schemas.data import DataConfigSchema
    from spectramr.infrastructure.training.strategies import diffusion

    assert "acceleration" not in DataConfigSchema.model_fields
    assert "base_acceleration" in AccelerationConfigSchema.model_fields
    # Positive assertion only: the removal left a comment quoting the old
    # `hasattr(self.config.data, "acceleration")` probe, so a negative grep
    # would match its own documentation.
    assert "accel_cfg.base_acceleration" in inspect.getsource(diffusion)


def test_validation_mask_fallback_reads_the_real_seed_field() -> None:
    """The deterministic-mask fallback read `self.config.training.seed`.

    Phase 4b moved the key to `run.seed`, and `training.seed` is a RAISE-posture
    rename — so `reject_renamed_keys("training")` blocks it in YAML *and* the
    attribute is absent on `TrainingStrategyConfigSchema`. There was no
    execution path on which the old spelling resolved. It only fires when a
    validation batch carries no pre-computed mask, i.e. the degraded path, so a
    happy-path smoke run never reached it.
    """
    import inspect

    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
    from spectramr.config.schemas.renames import RENAMES
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.training.strategies import diffusion

    assert "seed" not in TrainingStrategyConfigSchema.model_fields
    assert RENAMES["training.seed"].canonical == "run.seed"

    settings = TrainingSettings.settings_from_dict(
        {
            "config_version": CANONICAL_CONFIG_VERSION,
            "data": {"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
            "optimization": {"learning_rate": 1e-4},
            "logging": {},
            "model": {"model_type": "unet"},
            "run": {"seed": 1234},
        }
    )
    assert settings.run.seed == 1234
    # Positive assertion: the fix leaves a comment naming the old spelling, so a
    # negative grep would match the documentation of its own removal.
    assert "self.config.run.seed" in inspect.getsource(diffusion)


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
    mock._slice_to_target_contrast = lambda pred, target: (pred, target)
    mock._slice_to_target_contrast_single = lambda ksp: ksp
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
    assert "decompress_for_view" in called


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
    mock = MagicMock()
    mock.config = SimpleNamespace(
        data=SimpleNamespace(
            processing=SimpleNamespace(
                enable_kspace_normalization=True,
                enable_log_scaling=False,
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
        """The other half of the seam: undo ``log1p`` before scaling."""
        tree = self._tree()
        found = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "input_batch"
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "decompress_for_view"
        ]
        assert found, "``input_batch`` never passes through ``decompress_for_view``"


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
    from spectramr.core.metrics.computer import ValidationMetricsComputer
    from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig

    mock = MagicMock()
    mock.config = SimpleNamespace(
        data=SimpleNamespace(
            processing=SimpleNamespace(
                enable_kspace_normalization=False,
                enable_log_scaling=log_scaled,
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
        target = torch.rand(2, 2, 8, 8) * 50.0
        decompressed = decompress_for_view(compressed, log_scaled=True, channel_dim=1)
        assert not torch.allclose(decompressed, compressed), (
            "decompression was a no-op on this fixture, so the assertion below "
            "would hold whether or not the production path decompresses"
        )

        mock = _zf_mock(log_scaled=True, measurement=compressed)
        out = _run_validation_metrics(
            mock, torch.rand(2, 2, 8, 8) * 50.0, target, torch.rand(2, 2, 8, 8) * 50.0
        )
        assert out["val_zf_psnr"] == pytest.approx(_reference_psnr(decompressed, target), abs=1e-3)

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
        decompressed = [
            node.args[0].id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "decompress_for_view"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ]
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

        for call in (zf, primary):
            assert isinstance(call.args[1], ast.Name) and call.args[1].id == "target_transformed"
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
                losses=SimpleNamespace(reconstruction=SimpleNamespace(lambda_pre_dc_kspace=lam))
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
            members, torch.ones(2, 1, 1, 1), True, target, mock.config.validation, True
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
        from spectramr.infrastructure.validation.witness.checks.validation_metric_checks import (
            unresolved_metric_names,
        )

        emitted = DiffusionTrainingStrategy.capabilities.emitted_metrics
        names = [
            "empirical_coverage",
            "ensemble_std_mean",
            "val_empirical_coverage_8x",
            "val_empirical_coverage_mean",
            "val_ensemble_std_mean_32x",
            "val_ensemble_std_mean_mean",
        ]
        assert unresolved_metric_names(names, MetricsRegistry.is_registered, emitted) == []
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
